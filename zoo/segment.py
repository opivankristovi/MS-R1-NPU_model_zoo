"""Segmentation: YOLOv8-L-seg (COCO instance masks), U2-Net human cut-out, PP-HumanSeg, SAM 2.1 tiny (click-to-segment).

Masks are returned as uint8 arrays (0/255) of the original image size where useful; JSON results
carry only boxes/areas so they stay small.
"""
import threading

import cv2
import numpy as np

from .common import COCO80, color, hub, load_graph, nms, overlay_mask, softmax

SS = "ComputeVision/Semantic_Segmentation"
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _prep(img, size, mean=IMNET_MEAN, std=IMNET_STD):
    x = cv2.resize(img, size)[..., ::-1].astype(np.float32) / 255.0
    return ((x - np.array(mean, np.float32)) / np.array(std, np.float32)).transpose(2, 0, 1)[None]


def cutout(img, alpha):
    """BGR + 0..255 alpha -> BGRA (save as PNG for a transparent background)."""
    return np.dstack([img, alpha.astype(np.uint8)])


class YoloSeg:
    """YOLOv8-L instance segmentation, 640 letterbox, 80 COCO classes, 32 mask prototypes at 160x160."""

    def __init__(self):
        self.g = load_graph(f"{SS}/onnx_yolov8_segmentation/yolov8l_seg.cix")
        self.lock = threading.Lock()

    def __call__(self, img, conf=0.3, iou=0.45):
        from .common import letterbox
        H, W = img.shape[:2]
        lb, r, (pw, ph) = letterbox(img, 640)
        x = (lb[..., ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        with self.lock:
            p, protos = self.g.run([x])
        p = p.reshape(116, -1).T
        protos = protos.reshape(32, 160, 160)
        sc = p[:, 4:84].max(1)
        k = sc > conf
        p, sc = p[k], sc[k]
        cls = p[:, 4:84].argmax(1)
        b = p[:, :4].copy()
        b[:, :2] -= b[:, 2:4] / 2
        b[:, 2:4] += b[:, :2]
        keep = nms(b + cls[:, None] * 4096, sc, iou)  # class-aware NMS via offset
        dets, masks = [], []
        for i in keep:
            m = 1 / (1 + np.exp(-(p[i, 84:] @ protos.reshape(32, -1)))).reshape(160, 160)
            m = cv2.resize(m, (640, 640))[int(round(ph - 0.1)):int(round(640 - ph + 0.1)),
                                          int(round(pw - 0.1)):int(round(640 - pw + 0.1))]
            m = cv2.resize(m, (W, H)) > 0.5
            bx = ((b[i] - [pw, ph, pw, ph]) / r).clip(0, [W, H, W, H])
            x1, y1, x2, y2 = bx.astype(int)
            crop = np.zeros_like(m)
            crop[y1:y2, x1:x2] = m[y1:y2, x1:x2]
            masks.append(crop.astype(np.uint8) * 255)
            dets.append({"label": COCO80[cls[i]], "cls": int(cls[i]), "score": float(sc[i]),
                         "box": bx.tolist(), "area": int(crop.sum())})
        return {"detections": dets, "masks": masks}

    def draw(self, img, res):
        from .common import draw_boxes
        out = img.copy()
        for d, m in zip(res["detections"], res["masks"]):
            out = overlay_mask(out, m > 0, color(d["cls"]), 0.45)
        return draw_boxes(out, res["detections"])


class U2Net:
    """U2-Net human segmentation (320x320). Returns a soft alpha matte -> transparent PNG cut-out."""

    def __init__(self):
        self.g = load_graph(f"{SS}/onnx_u2net/u2net_human_seg.cix")
        self.lock = threading.Lock()

    def __call__(self, img):
        H, W = img.shape[:2]
        with self.lock:
            o = self.g.run([_prep(img, (320, 320))])[0].reshape(320, 320)
        o = (o - o.min()) / max(float(o.max() - o.min()), 1e-6)
        return {"alpha": (cv2.resize(o, (W, H)) * 255).astype(np.uint8)}

    def draw(self, img, res):
        return cutout(img, res["alpha"])  # BGRA


class PPHumanSeg:
    """PP-HumanSeg (192x192, very fast). Binary person mask."""

    def __init__(self):
        self.g = load_graph(f"{SS}/onnx_pphumanseg/pphuanseg.cix")
        self.lock = threading.Lock()

    def __call__(self, img, thresh=0.5):
        H, W = img.shape[:2]
        with self.lock:
            o = self.g.run([_prep(img, (192, 192))])[0].reshape(192, 192, 2)
        if o.min() < 0 or o.max() > 1.001:
            o = softmax(o, -1)
        p = cv2.resize(o[..., 1], (W, H))
        return {"alpha": (p * 255).astype(np.uint8), "mask_fraction": round(float((p > thresh).mean()), 4)}

    def draw(self, img, res):
        return overlay_mask(img, res["alpha"] > 127, (0, 255, 0), 0.45)


class Sam2:
    """SAM 2.1 tiny: image encoder (1024x1024) on the NPU (~once per image), prompt decoder on the CPU (ONNX).

    sam.set_image(img); masks = sam.segment(points=[(x, y, 1), (x2, y2, 0)])   # 1 = include, 0 = exclude
    or a box: sam.segment(box=(x1, y1, x2, y2)). Coordinates in original image pixels.
    """

    def __init__(self):
        import onnxruntime as ort
        self.enc = load_graph(f"{SS}/onnx_sam2_1_tiny/sam2_hiera_tiny_encoder_sim.cix")
        self.dec = ort.InferenceSession(hub(SS, "onnx_sam2_1_tiny/model/decoder.onnx"))
        self.in_names = [i.name for i in self.dec.get_inputs()]
        self.lock = threading.Lock()
        self.feats = self.size = None
        self._key = None

    def set_image(self, img):
        key = (img.shape, hash(img[::17, ::17].tobytes()))
        if key == self._key:
            return
        with self.lock:
            o = self.enc.run([_prep(img, (1024, 1024))])
        self.feats = (o[0].reshape(1, 32, 256, 256), o[1].reshape(1, 64, 128, 128), o[2].reshape(1, 256, 64, 64))
        self.size, self._key = img.shape[:2], key

    def segment(self, points=(), box=None):
        H, W = self.size
        coords, labels = [], []
        for x, y, *l in points:
            coords.append([x, y])
            labels.append(l[0] if l else 1)
        if box is not None:
            coords += [box[:2], box[2:]]
            labels += [2, 3]
        c = np.array(coords, np.float32)[None] * [1024 / W, 1024 / H]
        hr0, hr1, emb = self.feats
        feed = [emb, hr0, hr1, c.astype(np.float32), np.array(labels, np.float32)[None],
                np.zeros((1, 1, 256, 256), np.float32), np.array([0], np.float32), np.array([H, W], np.int32)]
        masks, scores = self.dec.run(None, dict(zip(self.in_names, feed)))[:2]
        masks, scores = masks.reshape(-1, H, W), scores.reshape(-1)
        best = int(scores.argmax())
        return (masks[best] > 0).astype(np.uint8) * 255, float(scores[best])

    def __call__(self, img, points="", box=""):
        """points: "x,y[,label]; x,y[,label]" (label 1=include, 0=exclude); box: "x1,y1,x2,y2".
        No prompt -> the image centre is used."""
        self.set_image(img)
        pts = [tuple(float(v) for v in p.split(",")) for p in str(points).split(";") if p.strip()]
        bx = [float(v) for v in str(box).split(",")] if str(box).strip() else None
        if not pts and bx is None:
            pts = [(img.shape[1] / 2, img.shape[0] / 2, 1)]
        mask, score = self.segment(pts, bx)
        return {"mask": mask, "score": round(score, 3), "points": pts, "box": bx,
                "area_fraction": round(float((mask > 0).mean()), 4)}

    def draw(self, img, res):
        out = overlay_mask(img, res["mask"] > 0, (255, 144, 30), 0.5)
        r = max(4, round(max(img.shape[:2]) / 150))
        for x, y, *l in res["points"]:
            cv2.circle(out, (int(x), int(y)), r, (0, 255, 0) if (not l or l[0]) else (0, 0, 255), -1)
        if res["box"]:
            x1, y1, x2, y2 = map(int, res["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return out
