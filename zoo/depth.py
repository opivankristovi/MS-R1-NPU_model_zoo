"""Monocular depth: Depth-Anything-V2 (518x518, best) and MiDaS v2 small (256x256, fast).

Both output RELATIVE inverse depth (disparity): larger = closer. No metric scale.
"""
import threading

import cv2
import numpy as np

from .common import colorize_depth, load_graph

DE = "ComputeVision/Depth_Estimation"


class DepthAnything:
    def __init__(self):
        self.g = load_graph(f"{DE}/onnx_depth_anything_v2/depth_anything_v2.cix")
        self.lock = threading.Lock()

    def __call__(self, img):
        H, W = img.shape[:2]
        x = cv2.resize(img, (518, 518), interpolation=cv2.INTER_CUBIC)[..., ::-1].astype(np.float32) / 255
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
        with self.lock:
            d = self.g.run([x])[0].reshape(518, 518)
        return {"depth": cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)}

    def draw(self, img, res):
        return colorize_depth(res["depth"])


class Midas:
    def __init__(self):
        self.g = load_graph(f"{DE}/onnx_MiDaS_v2/MiDaS_v2.cix")
        self.lock = threading.Lock()

    def __call__(self, img):
        H, W = img.shape[:2]
        x = cv2.resize(img[..., ::-1].astype(np.float32) / 255, (256, 256), interpolation=cv2.INTER_CUBIC)
        with self.lock:
            d = self.g.run([x.transpose(2, 0, 1)[None]])[0].reshape(256, 256)
        return {"depth": cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)}

    def draw(self, img, res):
        return colorize_depth(res["depth"])
