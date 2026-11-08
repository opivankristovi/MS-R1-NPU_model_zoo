"""Shared helpers: model paths, image I/O, letterbox, NMS, drawing.

Conventions used everywhere in the zoo:
  * images are OpenCV BGR uint8 arrays [H, W, 3]
  * boxes are absolute pixel xyxy floats in the ORIGINAL image
  * every model is a class with __call__(img, **params) -> dict  and  draw(img, result) -> BGR image
"""
import os
import time

import cv2
import numpy as np

from .npu import Graph

HUB = os.environ.get("ZOO_HUB", "/hub")  # = /mnt/data/npu-models/hub/models on the host
CV = f"{HUB}/ComputeVision"
GEN = f"{HUB}/Generative_AI"


def hub(*parts):
    p = os.path.join(HUB, *parts)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing - run tools/fetch_models.sh")
    with open(p, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            raise FileNotFoundError(f"{p} is still a git-lfs pointer - run tools/fetch_models.sh")
    return p


def load_graph(*parts, jobs=1):
    return Graph(hub(*parts), jobs=jobs)


# ---------------------------------------------------------------- image I/O
def imread(src):
    """Path, bytes, PIL image or ndarray -> BGR uint8 (EXIF rotation applied for files/bytes)."""
    if isinstance(src, np.ndarray):
        return src
    if hasattr(src, "convert"):  # PIL
        return cv2.cvtColor(np.asarray(src.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(src, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def to_pil(img_bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def encode(img_bgr, ext=".jpg", quality=90):
    ok, buf = cv2.imencode(ext, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else [])
    return buf.tobytes()


def letterbox(img, size=640, pad=114):
    """Keep aspect, pad to size x size (centered). Returns (padded, ratio, (pad_w, pad_h))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    pw, ph = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(ph - 0.1)), int(round(ph + 0.1))
    left, right = int(round(pw - 0.1)), int(round(pw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    return img, r, (pw, ph)


def topleft_pad(img, w, h, max_up=None):
    """Resize keeping aspect into the top-left of a w x h black canvas. Returns (canvas, scale).
    max_up caps upscaling of small images (face detectors miss faces that become huge)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    if max_up is not None:
        s = min(s, max_up)
    nw, nh = int(iw * s), int(ih * s)
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return canvas, s


def nchw(img, mean=0.0, std=1.0, rgb=True, scale=1 / 255.0):
    """HWC uint8 -> [1,3,H,W] float32, ((x*scale)-mean)/std."""
    x = img[..., ::-1] if rgb else img
    x = (x.astype(np.float32) * scale - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def nms(boxes, scores, iou=0.45):
    """boxes xyxy [N,4] -> kept indices (descending score)."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
    return np.array(keep, int).reshape(-1)


def xywh2xyxy(b):
    b = np.asarray(b, np.float32).copy()
    b[:, :2] -= b[:, 2:4] / 2
    b[:, 2:4] += b[:, :2]
    return b


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------- drawing
def color(i):
    rng = np.random.default_rng(int(i) * 7919 + 17)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_boxes(img, dets, thickness=None):
    """dets: list of {box:[x1,y1,x2,y2], label:str, score:float, [cls:int]}"""
    out = img.copy()
    t = thickness or max(2, round(max(img.shape[:2]) / 400))
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        c = color(d.get("cls", hash(d.get("label", "")) % 1000))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, t)
        txt = f"{d.get('label', '')} {d.get('score', 0):.2f}".strip()
        fs = t * 0.35
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        y = max(y1, th + 4)
        cv2.rectangle(out, (x1, y - th - 4), (x1 + tw + 4, y), c, -1)
        cv2.putText(out, txt, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(1, t // 2), cv2.LINE_AA)
    return out


def overlay_mask(img, mask, col=(0, 255, 0), alpha=0.5):
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
    return out


def colorize_depth(d):
    d = d.astype(np.float32)
    d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(s):
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                timer.t[name] = round((time.perf_counter() - s.t0) * 1000, 1)
        return _T()


COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]
