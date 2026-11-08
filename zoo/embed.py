"""Image/text embeddings: CLIP ViT-B/32 (both towers on NPU) and SigLIP so400m (image on NPU, text on CPU ONNX).

Use for zero-shot classification ("which of these labels fits?") and text->image / image->image search.
All embeddings are L2-normalised float32 vectors; similarity = dot product.
"""
import os
import re
import string
import threading

import cv2
import numpy as np

from .common import GEN, hub, load_graph, to_pil

CLIP_DIR = "Generative_AI/Image_to_Text/onnx_clip"
SIGLIP_DIR = "Generative_AI/Image_to_Text/onnx_siglip-so400m-patch14-384"


def _l2(v):
    v = np.asarray(v, np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


class Clip:
    """CLIP ViT-B/32, 512-d. ~17 ms per image or text on the NPU (int8: cosine 0.85-0.97 vs fp32).

    text_backend="cpu" runs the text tower as fp32 ONNX on the CPU instead (~30-60 ms, exact). Use it when
    the text vector feeds another network (YOLO-World needs it: int8 text embeddings ruin its scores)."""
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    SOT, EOT, CTX = 49406, 49407, 77

    def __init__(self, vision=True, text=True, text_backend="npu"):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        self.vis = load_graph(f"{CLIP_DIR}/clip_visual.cix") if vision else None
        self.txt = self.sess = None
        if text and text_backend == "npu":
            self.txt = load_graph(f"{CLIP_DIR}/clip_txt.cix")
        elif text:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ZOO_CPU_THREADS", "4"))
            self.sess = ort.InferenceSession(hub(CLIP_DIR, "model/clip_text_model_vitb32.onnx"), so)
        if text:
            self.tok = Tokenizer.from_file(hf_hub_download("openai/clip-vit-base-patch32", "tokenizer.json"))
            self.tok.no_padding()
            self.tok.no_truncation()
        self.lock = threading.Lock()

    def preprocess(self, img):
        h, w = img.shape[:2]
        s = 224 / min(w, h)
        pil = to_pil(img).resize((max(224, round(w * s)), max(224, round(h * s))), 3)  # bicubic
        w, h = pil.size
        l, t = (w - 224) // 2, (h - 224) // 2
        x = (np.asarray(pil.crop((l, t, l + 224, t + 224)), np.float32) / 255 - self.MEAN) / self.STD
        return x.transpose(2, 0, 1)[None]

    def embed_image(self, img):
        with self.lock:
            return _l2(self.vis.run([self.preprocess(img)])[0])

    def tokens(self, text):
        ids = self.tok.encode(text.lower()).ids
        if ids[:1] != [self.SOT]:
            ids = [self.SOT] + ids
        if ids[-1:] != [self.EOT]:
            ids = ids + [self.EOT]
        if len(ids) > self.CTX:
            ids = ids[: self.CTX - 1] + [self.EOT]
        return np.array(ids + [0] * (self.CTX - len(ids)), np.int32)[None]

    def embed_text(self, text):
        if self.sess is not None:
            return _l2(self.sess.run(None, {"TEXT": self.tokens(text)})[0])
        with self.lock:
            return _l2(self.txt.run([self.tokens(text)])[0])

    def __call__(self, img, labels="a photo of a cat, a photo of a dog"):
        """Zero-shot classification: softmax over cosine*100 against the comma-separated labels."""
        labels = [l.strip() for l in labels.split(",") if l.strip()] if isinstance(labels, str) else list(labels)
        v = self.embed_image(img)
        sims = np.array([float(v @ self.embed_text(l)) for l in labels])
        p = np.exp((sims - sims.max()) * 100)
        p /= p.sum()
        return {"labels": [{"label": l, "cosine": round(float(s), 4), "prob": round(float(q), 4)}
                           for l, s, q in sorted(zip(labels, sims, p), key=lambda t: -t[2])]}

    def draw(self, img, res):
        return _draw_labels(img, [f"{r['label']}: {r['prob']:.0%}" for r in res["labels"][:5]])


class Siglip:
    """SigLIP so400m-patch14-384, 1152-d. Image tower on the NPU; text tower is fp32 ONNX on the CPU
    (1.8 GB, loaded on first text call). Better retrieval/zero-shot than CLIP B/32, slower.
    Probabilities are independent sigmoids (labels don't compete), as SigLIP was trained."""
    LOGIT_SCALE, LOGIT_BIAS = float(np.exp(4.7215)), -16.5464
    CTX = 64

    def __init__(self, vision=True, text=True):
        self.vis = load_graph(f"{SIGLIP_DIR}/siglip_img.cix") if vision else None
        self.want_text = text
        self.sess = self.sp = None
        self.lock = threading.Lock()

    def _text_init(self):
        if self.sess is None:
            import onnxruntime as ort
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=hub(SIGLIP_DIR, "siglip-so400m-patch14-384/spiece.model"))
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ZOO_CPU_THREADS", "4"))
            self.sess = ort.InferenceSession(hub(SIGLIP_DIR, "model/siglip-so400m-patch14-384-txt.onnx"), so)

    def preprocess(self, img):
        x = cv2.resize(img, (384, 384), interpolation=cv2.INTER_CUBIC)
        return ((x[..., ::-1].astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]

    def embed_image(self, img):
        with self.lock:
            return _l2(self.vis.run([self.preprocess(img)])[0])

    def tokens(self, text):
        # SiglipTokenizer: lowercase, strip punctuation, collapse spaces, append </s>(1), pad with </s>(1) to 64
        t = text.lower().translate(str.maketrans("", "", string.punctuation))
        t = re.sub(r"\s+", " ", t).strip()
        ids = self.sp.encode(t)[: self.CTX - 1] + [1]
        return np.array(ids + [1] * (self.CTX - len(ids)), np.int64)[None]

    def embed_text(self, text):
        self._text_init()
        inp = self.sess.get_inputs()[0]
        ids = self.tokens(text)
        if "int32" in inp.type:
            ids = ids.astype(np.int32)
        return _l2(self.sess.run(None, {inp.name: ids})[0])

    def __call__(self, img, labels="a photo of a cat, a photo of a dog"):
        labels = [l.strip() for l in labels.split(",") if l.strip()] if isinstance(labels, str) else list(labels)
        v = self.embed_image(img)
        sims = np.array([float(v @ self.embed_text(l)) for l in labels])
        p = 1 / (1 + np.exp(-(sims * self.LOGIT_SCALE + self.LOGIT_BIAS)))
        return {"labels": [{"label": l, "cosine": round(float(s), 4), "prob": round(float(q), 4)}
                           for l, s, q in sorted(zip(labels, sims, p), key=lambda t: -t[1])]}

    def draw(self, img, res):
        return _draw_labels(img, [f"{r['label']}: {r['prob']:.0%}" for r in res["labels"][:5]])


def _draw_labels(img, lines):
    out = img.copy()
    fs = max(0.5, out.shape[1] / 900)
    th = int(28 * fs)
    cv2.rectangle(out, (0, 0), (out.shape[1], th * len(lines) + 10), (0, 0, 0), -1)
    for i, l in enumerate(lines):
        cv2.putText(out, l, (8, th * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(fs * 2)), cv2.LINE_AA)
    return out
