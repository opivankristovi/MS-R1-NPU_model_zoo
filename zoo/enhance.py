"""Image enhancement: super-resolution (Real-ESRGAN x4, ESPCN x3, VDSR) and low-light / night-haze fixes.

All the hub models have small FIXED input sizes (224-512 px). To work on real photos:
  * super-resolution runs on overlapping tiles that are stitched back (Tiler)
  * enhancement runs on a downscaled copy; the result is upsampled and the full-res detail
    of the original is added back (detail transfer), so the output keeps the input's sharpness.
"""
import threading

import cv2
import numpy as np

from .common import load_graph

SR = "ComputeVision/Super_Resolution"
IE = "ComputeVision/Image_Enhancement"


def tiled(img, fn, tile, scale, overlap=16, channels=3):
    """Run fn(tile_img[tile,tile,(C)]) -> [tile*scale, tile*scale, (C)] float over img with overlap, blended."""
    H, W = img.shape[:2]
    ph, pw = max(H, tile), max(W, tile)
    if (ph, pw) != (H, W):
        img = cv2.copyMakeBorder(img, 0, ph - H, 0, pw - W, cv2.BORDER_REFLECT)
    shape = (ph * scale, pw * scale) + ((channels,) if channels > 1 else ())
    acc, wsum = np.zeros(shape, np.float32), np.zeros((ph * scale, pw * scale), np.float32)
    step = tile - 2 * overlap
    ramp = np.minimum(np.arange(tile * scale) + 1, (tile * scale - np.arange(tile * scale)))
    wt = np.minimum(np.minimum.outer(ramp, ramp), overlap * scale).astype(np.float32) + 1e-3
    ys = list(range(0, max(ph - tile, 0) + 1, step))
    xs = list(range(0, max(pw - tile, 0) + 1, step))
    if ys[-1] != ph - tile:
        ys.append(ph - tile)
    if xs[-1] != pw - tile:
        xs.append(pw - tile)
    for y in ys:
        for x in xs:
            out = fn(img[y:y + tile, x:x + tile])
            sl = (slice(y * scale, (y + tile) * scale), slice(x * scale, (x + tile) * scale))
            acc[sl] += out * (wt[..., None] if channels > 1 else wt)
            wsum[sl] += wt
    acc /= wsum[..., None] if channels > 1 else wsum
    return acc[:H * scale, :W * scale]


def detail_transfer(img, small_out):
    """Upsample a low-res enhanced image to img's size and add img's high-frequency detail back."""
    H, W = img.shape[:2]
    h, w = small_out.shape[:2]
    up = cv2.resize(small_out.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    base = cv2.resize(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32), (W, H),
                      interpolation=cv2.INTER_CUBIC)
    # scale the detail by the local brightening so dark-image noise isn't amplified unevenly
    gain = np.clip((up.mean(2, keepdims=True) + 8) / (base.mean(2, keepdims=True) + 8), 0.5, 4)
    return np.clip(up + (img.astype(np.float32) - base) * gain, 0, 255).astype(np.uint8)


def _ycc_merge(y_big, img, scale):
    H, W = img.shape[:2]
    ycc = cv2.cvtColor(cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = np.clip(y_big, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


class _Upscaler:
    max_side = 1024  # input limit (output is 3-4x bigger)

    def _limit(self, img):
        s = self.max_side / max(img.shape[:2])
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img

    def draw(self, img, res):
        return res["image"]


class RealEsrgan(_Upscaler):
    """Real-ESRGAN x4plus: 4x upscaling that also removes JPEG/blur artefacts. 224 px tiles."""

    def __init__(self):
        self.g = load_graph(f"{SR}/onnx_real_esrgan/real_esrgan.cix")
        self.lock = threading.Lock()

    def _tile(self, t):
        x = (t.astype(np.float32) / 255).transpose(2, 0, 1)[None]  # BGR, as the hub calibrated
        with self.lock:
            o = self.g.run([x])[0].reshape(3, 896, 896)
        return o.transpose(1, 2, 0) * 255

    def __call__(self, img):
        img = self._limit(img)
        return {"image": np.clip(tiled(img, self._tile, 224, 4), 0, 255).astype(np.uint8)}


class Espcn(_Upscaler):
    """ESPCN 'super_resolution_10' (ONNX model zoo): fast 3x upscaling of the luma channel."""

    def __init__(self):
        self.g = load_graph(f"{SR}/onnx_super_resolution/super_resolution_10.cix")
        self.lock = threading.Lock()

    def _tile(self, y):
        with self.lock:
            o = self.g.run([(y.astype(np.float32) / 255)[None, None]])[0].reshape(672, 672)
        return o * 255

    def __call__(self, img):
        img = self._limit(img)
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0]
        return {"image": _ycc_merge(tiled(y, self._tile, 224, 3, channels=1), img, 3)}


class Vdsr(_Upscaler):
    """VDSR: refines a bicubic-upscaled luma channel (any scale; default 2x). 256 px tiles."""

    def __init__(self):
        self.g = load_graph(f"{SR}/onnx_vdsr/vdsr.cix")
        self.lock = threading.Lock()

    def _tile(self, y):
        with self.lock:
            o = self.g.run([(y.astype(np.float32) / 255)[None, None]])[0].reshape(256, 256)
        return o * 255

    def __call__(self, img, scale=2):
        img = self._limit(img)
        scale = int(scale)
        H, W = img.shape[:2]
        up = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)
        y = cv2.cvtColor(up, cv2.COLOR_BGR2YCrCb)
        y[..., 0] = np.clip(tiled(y[..., 0], self._tile, 256, 1, channels=1), 0, 255).astype(np.uint8)
        return {"image": cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)}


class Agllnet(_Upscaler):
    """AGLLNet low-light enhancement (256x256 core + detail transfer)."""

    def __init__(self):
        self.g = load_graph(f"{IE}/onnx_AGLLNet/AGLLNet.cix")
        self.lock = threading.Lock()

    def __call__(self, img):
        x = (cv2.resize(img, (256, 256))[..., ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None]
        with self.lock:
            o = self.g.run([x])[0].reshape(256, 256, 10)[..., 4:7]
        small = np.clip(o * 255, 0, 255)[..., ::-1]
        return {"image": detail_transfer(img, small)}


class Pairlie(_Upscaler):
    """PairLIE low-light enhancement (512x512 core + detail transfer). exposure ~0.3-0.7."""

    def __init__(self):
        self.g = load_graph(f"{IE}/onnx_pairlie/pairlie.cix")
        self.lock = threading.Lock()

    def __call__(self, img, exposure=0.5):
        x = (cv2.resize(img, (512, 512))[..., ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None]
        with self.lock:
            o = self.g.run([x, np.array([exposure], np.float32)])[0].reshape(3, 512, 512)
        small = np.clip(o.transpose(1, 2, 0) * 255, 0, 255)[..., ::-1]
        return {"image": detail_transfer(img, small)}


class NightDehaze(_Upscaler):
    """Night-time dehazing (glow/haze around lights), 512x512 core + detail transfer."""

    def __init__(self):
        self.g = load_graph(f"{IE}/onnx_nighttime_dehaze/nighttime_dehaze.cix")
        self.lock = threading.Lock()

    def __call__(self, img):
        x = ((cv2.resize(img, (512, 512))[..., ::-1].astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        with self.lock:
            o = self.g.run([x])[0].reshape(3, 512, 512).transpose(1, 2, 0)
        if o.max() <= 1.5:  # model variant emitting 0..1 instead of 0..255
            o = o * 255
        return {"image": detail_transfer(img, np.clip(o, 0, 255)[..., ::-1])}
