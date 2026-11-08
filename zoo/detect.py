"""Object detection: YOLOv8-L / YOLOv12-M / YOLOX-M/L (COCO-80) and YOLO-World v2 (open vocabulary)."""
import threading

import cv2
import numpy as np

from .common import COCO80, CV, draw_boxes, load_graph, nms, xywh2xyxy

OD = "ComputeVision/Object_Detection"
S = 640


def _classwise(boxes, scores, cls, iou, names, sx, sy, max_det=300):
    out = []
    for c in np.unique(cls):
        m = np.where(cls == c)[0]
        for i in m[nms(boxes[m], scores[m], iou)]:
            x1, y1, x2, y2 = boxes[i]
            out.append({"label": names[c], "cls": int(c), "score": float(scores[i]),
                        "box": [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)]})
    out.sort(key=lambda d: -d["score"])
    return out[:max_det]


class _Det:
    names = COCO80

    def draw(self, img, res):
        return draw_boxes(img, res["detections"])


class Yolov8(_Det):
    """YOLOv8-L or YOLOv12-M. Input: plain resize to 640, BGR/255 (as the hub calibrated it). Output [84,8400]."""
    FILES = {"yolov8_l": f"{OD}/onnx_yolov8_l/yolov8_l.cix", "yolov12_m": f"{OD}/onnx_yolov12_m/yolov12_m.cix"}

    def __init__(self, variant="yolov8_l"):
        self.g = load_graph(self.FILES[variant])
        self.lock = threading.Lock()

    def __call__(self, img, conf=0.3, iou=0.45):
        h, w = img.shape[:2]
        x = cv2.resize(img, (S, S)).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        with self.lock:
            p = self.g.run([x])[0].reshape(84, -1).T
        sc = p[:, 4:].max(1)
        k = sc > conf
        p, sc = p[k], sc[k]
        return {"detections": _classwise(xywh2xyxy(p[:, :4]), sc, p[:, 4:].argmax(1), iou, self.names, w / S, h / S)}


class Yolox(_Det):
    """YOLOX-M / YOLOX-L. Input: plain resize 640, BGR uint8 0..255. Output [8400,85] (grid-relative)."""
    FILES = {"yolox_m": f"{OD}/onnx_yolox_m/yolox_m.cix", "yolox_l": f"{OD}/onnx_yolox_l/yolox_l.cix"}

    def __init__(self, variant="yolox_m"):
        self.g = load_graph(self.FILES[variant])
        self.lock = threading.Lock()
        g, st = [], []
        for s in (8, 16, 32):
            n = S // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g.append(np.stack((xv, yv), 2).reshape(-1, 2))
            st.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(g).astype(np.float32), np.concatenate(st).astype(np.float32)

    def __call__(self, img, conf=0.35, iou=0.45):
        h, w = img.shape[:2]
        x = cv2.resize(img, (S, S)).astype(np.float32).transpose(2, 0, 1)[None]
        with self.lock:
            o = self.g.run([x])[0].reshape(-1, 85)
        xy = (o[:, :2] + self.grid) * self.stride
        wh = np.exp(np.clip(o[:, 2:4], -10, 10)) * self.stride
        s = o[:, 4:5] * o[:, 5:]
        cls = s.argmax(1)
        sc = s[np.arange(len(cls)), cls]
        k = sc > conf
        boxes = np.c_[xy - wh / 2, xy + wh / 2][k]
        return {"detections": _classwise(boxes, sc[k], cls[k], iou, self.names, w / S, h / S)}


class YoloWorld(_Det):
    """YOLOv8-L-World v2: detect anything you can name. Up to 80 class prompts per call.

    Class prompts are embedded with CLIP ViT-B/32 text - the encoder YOLO-World was trained with - as fp32
    ONNX on the CPU (the int8 NPU text tower is NOT accurate enough here: "person" scores 0.03 instead of 0.82).
    Embeddings are cached per prompt.
    """
    MAXC = 80

    def __init__(self):
        from .embed import Clip
        self.g = load_graph(f"{OD}/onnx_yolov8l_worldv2/yolov8l-worldv2.cix")
        self.clip = Clip(vision=False, text_backend="cpu")
        self.cache = {}
        self.lock = threading.Lock()

    def embed(self, classes):
        e = np.zeros((1, self.MAXC, 512), np.float32)
        for i, c in enumerate(classes[: self.MAXC]):
            if c not in self.cache:
                self.cache[c] = self.clip.embed_text(c)
            e[0, i] = self.cache[c]
        e[e == 0] = 0.005  # as the hub does for padding rows
        return e

    def _pass(self, canvas, e, n):
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        with self.lock:
            return self.g.run([x, e])[0].reshape(4 + self.MAXC, -1).T[:, : 4 + n]

    def __call__(self, img, classes=("person",), conf=0.25, iou=0.45, multiscale=True):
        """multiscale: the hub .cix's stride-32 (large object) head is broken by quantization (scores ~0.001),
        so objects larger than ~1/3 of the frame are missed. A second pass with the image shrunk to 1/3 in the
        top-left of the 640 canvas makes them small enough for the working heads; passes are merged with NMS.
        (Measured on a full-frame truck: score 0.01 at 1/1 and 1/2 scale, 0.62-0.82 at 1/3.)"""
        if isinstance(classes, str):
            classes = [c.strip() for c in classes.split(",") if c.strip()]
        classes = list(classes)[: self.MAXC]
        n = len(classes)
        h, w = img.shape[:2]
        e = self.embed(classes)
        p = self._pass(cv2.resize(img, (S, S)), e, n)
        b = xywh2xyxy(p[:, :4]) * [w / S, h / S, w / S, h / S]
        boxes, scores = [b], [p[:, 4:]]
        if multiscale:
            q = S // 3
            canvas = np.zeros((S, S, 3), np.uint8)
            canvas[:q, :q] = cv2.resize(img, (q, q))
            p2 = self._pass(canvas, e, n)
            b2 = xywh2xyxy(p2[:, :4]) * [w / q, h / q, w / q, h / q]
            boxes.append(b2)
            scores.append(p2[:, 4:])
        b, sc = np.concatenate(boxes).clip(0, [w, h, w, h]), np.concatenate(scores)
        best, cls = sc.max(1), sc.argmax(1)
        k = best > conf
        return {"detections": _classwise(b[k], best[k], cls[k], iou, classes, 1.0, 1.0)}
