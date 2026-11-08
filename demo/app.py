"""Model zoo demo: pick a model, upload an image (or a sample), see the drawn result + JSON + timings.

Run:  docker compose up -d demo   ->  http://<board>:8097
API:  GET  /api/models                      registry (task, description, params, loaded)
      GET  /api/samples, /samples/<name>     sample images
      POST /api/run/<model>  multipart: image=<file> | sample=<name>, ref=<file> (face_id), + model params
           -> {"result": {...}, "ms": {"decode", "infer", "draw"}, "image": "data:image/...;base64,..."}
"""
import base64
import os
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

import zoo
from zoo.common import encode, imread
from zoo.worker import ModelWorker

SAMPLES = os.environ.get("ZOO_SAMPLES", "/samples")
MAX_SIDE = int(os.environ.get("ZOO_MAX_SIDE", "2048"))   # downscale huge uploads

app = FastAPI(title="NPU model zoo")
HERE = os.path.dirname(__file__)


worker = ModelWorker()  # models live in a child process that is replaced when NPU memory runs out


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not isinstance(v, np.ndarray)}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x if not isinstance(v, np.ndarray)]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _shrink(img):
    import cv2
    s = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()


@app.get("/api/models")
def models():
    return {k: {"task": v["task"], "desc": v["desc"], "loaded": k in worker.loaded,
                "params": {p: {"type": t, "default": d} for p, (t, d) in v["params"].items()}}
            for k, v in zoo.MODELS.items()}


@app.get("/api/samples")
def samples():
    return sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".jpeg", ".png")))


@app.get("/samples/{name}")
def sample(name: str):
    p = os.path.join(SAMPLES, os.path.basename(name))
    if not os.path.isfile(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/run/{name}")
async def run(name: str, request: Request, image: UploadFile | None = File(None), sample: str = Form(""),
              ref: UploadFile | None = File(None)):
    if name not in zoo.MODELS:
        raise HTTPException(404, f"unknown model {name}")
    form = await request.form()
    spec = zoo.MODELS[name]
    t = {}
    t0 = time.perf_counter()
    if image is not None and image.filename:
        img = imread(await image.read())
    elif sample:
        img = imread(os.path.join(SAMPLES, os.path.basename(sample)))
    else:
        raise HTTPException(400, "send image=<file> or sample=<name>")
    img = _shrink(img)
    kwargs = {}
    for p, (typ, default) in spec["params"].items():
        if typ == "image":
            if p == "ref" and ref is not None and ref.filename:
                kwargs[p] = _shrink(imread(await ref.read()))
            continue
        v = form.get(p)
        if v is None or v == "":
            continue
        kwargs[p] = {"float": float, "int": int}.get(typ, str)(v)
    t["decode"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    try:
        res, vis = worker.call(name, img, **kwargs)  # includes model load on first use, and drawing
    except Exception as e:  # surface model errors to the UI
        raise HTTPException(500, str(e))
    t["infer"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    png = vis.ndim == 3 and vis.shape[2] == 4 or spec["task"] == "enhance"
    data = encode(vis, ".png" if png else ".jpg")
    t["draw"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"model": name, "worker_restarts": worker.restarts, "size": [img.shape[1], img.shape[0]], "out_size": [vis.shape[1], vis.shape[0]],
            "ms": t, "result": _jsonable(res),
            "image": f"data:image/{'png' if png else 'jpeg'};base64," + base64.b64encode(data).decode()}
