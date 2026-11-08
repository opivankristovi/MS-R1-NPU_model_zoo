"""TEMPLATE app: text-prompted cut-out = YOLO-World (find "a red mug") -> SAM2 (mask each box).

Copy this folder to apps/<your_app>/, rename, and replace the logic. It shows the conventions:
  * models come from the zoo registry (zoo.get) or the classes directly (zoo.detect.YoloWorld ...)
  * load models once at startup, reuse them for every request (loading costs 0.1-3 s and NPU memory)
  * images are BGR uint8 numpy arrays (zoo.common.imread / encode)
  * NPU calls are serialized per model by the model's own lock; FastAPI's threadpool is fine

Run (from the repo root):  docker compose -f apps/_template/compose.yml up --build
     curl -F image=@samples/truck.jpg -F prompt=truck localhost:8098/cutout -o cut.png
"""
import io

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

import zoo
from zoo.common import encode, imread

app = FastAPI(title="cutout (zoo template)")
world = zoo.get("yolo_world")
sam = zoo.get("sam2")


def cutout(img, prompt, conf=0.25):
    dets = world(img, classes=prompt, conf=conf)["detections"]
    alpha = np.zeros(img.shape[:2], np.uint8)
    sam.set_image(img)                      # NPU encoder once per image
    H, W = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.08  # detector boxes can be tight/coarse: pad a little
        box = (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))
        centre = ((x1 + x2) / 2, (y1 + y2) / 2, 1)   # + a positive point steadies SAM on big objects
        # a box covering most of the frame is no constraint for SAM (it grabs background): point only then
        whole = (box[2] - box[0]) * (box[3] - box[1]) > 0.5 * W * H
        mask, _ = sam.segment(points=[centre], box=None if whole else box)  # CPU decoder (~50 ms)
        alpha |= mask
    return dets, np.dstack([img, alpha])


@app.post("/cutout")
async def cutout_endpoint(image: UploadFile = File(...), prompt: str = Form(...), conf: float = Form(0.25)):
    dets, rgba = cutout(imread(await image.read()), prompt, conf)
    return Response(encode(rgba, ".png"), media_type="image/png",
                    headers={"X-Detections": str(len(dets))})


@app.get("/health")
def health():
    return {"ok": True, "models": zoo.loaded()}
