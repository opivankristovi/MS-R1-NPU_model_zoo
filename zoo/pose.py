"""Pose: YOLOv8s-pose (multi-person, 17 COCO keypoints) and a 21-point hand landmark model."""
import threading

import cv2
import numpy as np

from .common import draw_boxes, letterbox, load_graph, nms

PE = "ComputeVision/Pose_Estimation"
KPT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
             "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


class YoloPose:
    """YOLOv8s-pose: persons with 17 keypoints [x, y, conf] in image pixels."""

    def __init__(self):
        self.g = load_graph(f"{PE}/onnx_yolov8s_pose/yolov8s-pose.cix")
        self.lock = threading.Lock()

    def __call__(self, img, conf=0.4, iou=0.45):
        H, W = img.shape[:2]
        lb, r, (pw, ph) = letterbox(img, 640)
        x = (lb[..., ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        with self.lock:
            p = self.g.run([x])[0].reshape(56, -1).T
        p = p[p[:, 4] > conf]
        b = p[:, :4].copy()
        b[:, :2] -= b[:, 2:4] / 2
        b[:, 2:4] += b[:, :2]
        people = []
        for i in nms(b, p[:, 4], iou):
            bx = ((b[i] - [pw, ph, pw, ph]) / r).clip(0, [W, H, W, H])
            k = p[i, 5:].reshape(17, 3).copy()
            k[:, 0] = (k[:, 0] - pw) / r
            k[:, 1] = (k[:, 1] - ph) / r
            people.append({"box": bx.tolist(), "score": float(p[i, 4]), "keypoints": k.round(2).tolist()})
        return {"people": people}

    def draw(self, img, res, kconf=0.5):
        out = draw_boxes(img, [{"box": p["box"], "score": p["score"], "label": "person", "cls": 0} for p in res["people"]])
        t = max(2, round(max(img.shape[:2]) / 300))
        for p in res["people"]:
            k = np.array(p["keypoints"])
            for a, b in SKELETON:
                if k[a, 2] > kconf and k[b, 2] > kconf:
                    cv2.line(out, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), (255, 200, 0), t, cv2.LINE_AA)
            for x, y, c in k:
                if c > kconf:
                    cv2.circle(out, (int(x), int(y)), t + 2, (0, 0, 255), -1)
        return out


class HandPose:
    """21 hand landmarks (ResNet-50, 256x256). Expects an image CROPPED to one hand
    (e.g. a box from YOLO-World with class "hand"); returns points in the input image's pixels."""

    def __init__(self):
        self.g = load_graph(f"{PE}/onnx_handpose/handpose_resnet50_size256.cix")
        self.lock = threading.Lock()

    def __call__(self, img):
        H, W = img.shape[:2]
        x = (cv2.resize(img, (256, 256))[..., ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        with self.lock:
            o = self.g.run([x])[0].reshape(21, 2)
        return {"points": (o * [W, H]).round(1).tolist()}

    def draw(self, img, res):
        out = img.copy()
        pts = np.array(res["points"]).astype(int)
        t = max(2, round(max(img.shape[:2]) / 200))
        for i, (a, b) in enumerate(HAND_EDGES):
            col = [(0, 215, 255), (255, 115, 55), (5, 255, 55), (25, 15, 255), (225, 15, 55)][i // 4]
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), col, t, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, tuple(p), t + 1, (255, 255, 255), -1)
        return out
