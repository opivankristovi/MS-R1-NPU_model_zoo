"""Faces: SCRFD-10G detection (+5 landmarks), ArcFace R50 embeddings, CenterFace, RFB-320, age/gender.

Typical use (face search / recognition):
    fr = FaceRecognizer()
    faces = fr.faces(img)               # [{box, score, kps, emb(512, L2)}]
    sim = faces[0]["emb"] @ other["emb"] # same person when > ~0.35-0.4
"""
import threading

import cv2
import numpy as np

from .common import draw_boxes, load_graph, nms, softmax, topleft_pad

FD = "ComputeVision/Face_Detection"
FR = "ComputeVision/Face_Recognition/onnx_scrfd_arcface"
BA = "ComputeVision/Body_Analysis"

ARC_REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def align(img, kps, size=112):
    """Similarity-warp a face to the ArcFace 112x112 template using its 5 landmarks."""
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), ARC_REF * size / 112, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (size, size), borderValue=0)


def _draw_faces(img, faces, label_fn=lambda f: "face"):
    out = draw_boxes(img, [{"box": f["box"], "score": f["score"], "label": label_fn(f), "cls": 1} for f in faces])
    r = max(2, round(max(img.shape[:2]) / 400))
    for f in faces:
        for x, y in f.get("kps", []):
            cv2.circle(out, (int(x), int(y)), r, (0, 0, 255), -1)
    return out


class Scrfd:
    """SCRFD-10G face detector, 640x640 input, 3 strides x 2 anchors, 5 landmarks."""
    S = 640

    def __init__(self):
        self.g = load_graph(f"{FR}/scrfd.cix")
        self.lock = threading.Lock()
        self.centers = {}
        for st in (8, 16, 32):
            n = self.S // st
            c = np.stack(np.mgrid[:n, :n][::-1], -1).reshape(-1, 2).astype(np.float32) * st
            self.centers[st] = np.repeat(c, 2, axis=0)

    def __call__(self, img, conf=0.5, iou=0.4):
        canvas, s = topleft_pad(img, self.S, self.S, max_up=1.0)
        x = ((canvas[..., ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        with self.lock:
            o = self.g.run([x])
        boxes, scores, kps = [], [], []
        for i, st in enumerate((8, 16, 32)):
            sc = o[i].reshape(-1)
            bb = o[i + 3].reshape(-1, 4) * st
            kp = o[i + 6].reshape(-1, 10) * st
            c = self.centers[st]
            k = sc >= conf
            c, bb, kp = c[k], bb[k], kp[k]
            boxes.append(np.c_[c - bb[:, :2], c + bb[:, 2:]])
            kps.append((np.tile(c, 5) + kp).reshape(-1, 5, 2))
            scores.append(sc[k])
        boxes, scores, kps = np.concatenate(boxes) / s, np.concatenate(scores), np.concatenate(kps) / s
        keep = nms(boxes, scores, iou)
        faces = [{"box": boxes[i].tolist(), "score": float(scores[i]), "kps": kps[i].tolist()} for i in keep]
        return {"faces": faces}

    def draw(self, img, res):
        return _draw_faces(img, res["faces"])


class ArcFace:
    """ArcFace ResNet-50 (WebFace600K), aligned 112x112 RGB face -> 512-d L2-normalised embedding."""

    def __init__(self):
        self.g = load_graph(f"{FR}/arcface.cix")
        self.lock = threading.Lock()

    def embed(self, face112_bgr):
        x = ((face112_bgr[..., ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        with self.lock:
            v = self.g.run([x])[0]
        return v / max(float(np.linalg.norm(v)), 1e-12)


class FaceRecognizer:
    """SCRFD + ArcFace. __call__(img, ref=<image>) marks each face with its similarity to the ref's main face."""
    THRESH = 0.35

    def __init__(self):
        self.det, self.arc = Scrfd(), ArcFace()

    def faces(self, img, conf=0.5, min_size=20):
        out = []
        for f in self.det(img, conf=conf)["faces"]:
            x1, y1, x2, y2 = f["box"]
            if min(x2 - x1, y2 - y1) < min_size:
                continue
            f["emb"] = self.arc.embed(align(img, f["kps"]))
            out.append(f)
        return out

    def __call__(self, img, ref=None, conf=0.5):
        faces = self.faces(img, conf)
        if ref is not None:
            rf = self.faces(ref, conf)
            if rf:
                main = max(rf, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
                for f in faces:
                    f["similarity"] = round(float(f["emb"] @ main["emb"]), 3)
                    f["match"] = f["similarity"] > self.THRESH
        for f in faces:
            f["emb"] = f["emb"].round(5).tolist()
        return {"faces": faces}

    def draw(self, img, res):
        return _draw_faces(img, res["faces"], lambda f: (f"sim {f['similarity']:.2f}" + (" MATCH" if f["match"] else ""))
                           if "similarity" in f else "face")


class CenterFace:
    """CenterFace: anchor-free, 640x640 (top-left padded), heatmap/scale/offset/landmarks at stride 4.
    Note: confidence saturates at ~0.21 on this int8 build - use it as a yes/no detector."""
    S = 640

    def __init__(self):
        self.g = load_graph(f"{FD}/onnx_centerface/centerface.cix")
        self.lock = threading.Lock()

    def __call__(self, img, conf=0.2, iou=0.3):
        # The .cix takes raw RGB 0..255 (uint8 input, scale 1) and its int8 heatmap saturates at ~0.214,
        # so scores are not meaningful above 0.2; the hub uses threshold 0.2 too.
        canvas, s = topleft_pad(img, self.S, self.S, max_up=1.0)
        x = canvas[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None]
        with self.lock:
            hm, sc, off, lm = self.g.run([x])
        n = self.S // 4
        hm, sc, off, lm = hm.reshape(n, n), sc.reshape(2, n, n), off.reshape(2, n, n), lm.reshape(10, n, n)
        ys, xs = np.where(hm > conf)
        boxes, scores, kps = [], [], []
        for y, x_ in zip(ys, xs):
            h, w = np.exp(sc[0, y, x_]) * 4, np.exp(sc[1, y, x_]) * 4
            cx, cy = (x_ + off[1, y, x_] + 0.5) * 4, (y + off[0, y, x_] + 0.5) * 4
            x1, y1 = max(0, cx - w / 2), max(0, cy - h / 2)
            boxes.append([x1, y1, min(x1 + w, self.S), min(y1 + h, self.S)])
            scores.append(hm[y, x_])
            kps.append([[lm[j * 2 + 1, y, x_] * w + x1, lm[j * 2, y, x_] * h + y1] for j in range(5)])
        if not boxes:
            return {"faces": []}
        boxes, scores, kps = np.array(boxes) / s, np.array(scores), np.array(kps) / s
        return {"faces": [{"box": boxes[i].tolist(), "score": float(scores[i]), "kps": kps[i].tolist()}
                          for i in nms(boxes, scores, iou)]}

    def draw(self, img, res):
        return _draw_faces(img, res["faces"])


class Rfb320:
    """Ultra-Light-Fast RFB-320: tiny face detector, 320x240 input, no landmarks. Fast, weaker on small faces."""

    def __init__(self):
        self.g = load_graph(f"{FD}/onnx_rfb_320/rfb_320.cix")
        self.lock = threading.Lock()

    def __call__(self, img, conf=0.7, iou=0.3):
        h, w = img.shape[:2]
        x = ((cv2.resize(img, (320, 240))[..., ::-1].astype(np.float32) - 127) / 128).transpose(2, 0, 1)[None]
        with self.lock:
            c, b = self.g.run([x])
        c, b = c.reshape(-1, 2), b.reshape(-1, 4)
        k = c[:, 1] > conf
        b, sc = b[k] * [w, h, w, h], c[k, 1]
        return {"faces": [{"box": b[i].tolist(), "score": float(sc[i])} for i in nms(b, sc, iou)]}

    def draw(self, img, res):
        return _draw_faces(img, res["faces"])


class AgeGender:
    """SCRFD faces -> GoogLeNet age bucket + gender (Levi & Hassner, Adience). Rough estimates only."""
    AGES = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
    GENDERS = ["male", "female"]

    def __init__(self):
        self.det = Scrfd()
        self.age = load_graph(f"{BA}/onnx_age_googlenet/age_googlenet.cix")
        self.gen = load_graph(f"{BA}/onnx_gender_googlenet/gender_googlenet.cix")
        self.lock = threading.Lock()

    @staticmethod
    def _prep(crop):
        x = cv2.resize(crop, (224, 224))[..., ::-1].astype(np.float32) - np.array([104, 117, 123], np.float32)
        return x.transpose(2, 0, 1)[None]

    def classify(self, face_crop):
        x = self._prep(face_crop)
        with self.lock:
            a, g = self.age.run([x])[0], self.gen.run([x])[0]
        if a.min() < 0 or abs(a.sum() - 1) > 0.05:
            a = softmax(a)
        if g.min() < 0 or abs(g.sum() - 1) > 0.05:
            g = softmax(g)
        return {"age": self.AGES[int(a.argmax())], "age_p": round(float(a.max()), 3),
                "gender": self.GENDERS[int(g.argmax())], "gender_p": round(float(g.max()), 3)}

    def __call__(self, img, conf=0.5, margin=0.4):
        H, W = img.shape[:2]
        faces = self.det(img, conf)["faces"]
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            mx, my = (x2 - x1) * margin, (y2 - y1) * margin
            crop = img[max(0, int(y1 - my)):min(H, int(y2 + my)), max(0, int(x1 - mx)):min(W, int(x2 + mx))]
            if crop.size:
                f.update(self.classify(crop))
        return {"faces": faces}

    def draw(self, img, res):
        return _draw_faces(img, res["faces"], lambda f: f"{f.get('gender', '?')} {f.get('age', '?')}")
