# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo turns the CIX P1 NPU (Zhouyi X2, `/dev/aipu`) into a toolbox of 26 tested vision models. Use it
to build applications: new folders under `apps/`, each a small Docker service. README.md has the model
table, timings and known model defects. Read it before choosing models.

Machine-specific notes (other services on the NPU, ports in use, cameras, sibling projects) live in
`CLAUDE.local.md`, which is not committed. Read it if it exists.

## Commands
There is no linter or unit-test framework; `tests/smoke.py` (real inference on `samples/`) is the test.
```bash
tools/fetch_runtime.sh                               # once; CIX NOE runtime into vendor/ (not committed)
tools/fetch_models.sh                                # once; idempotent, ~6 GB into $MODELS/hub
docker compose build                                 # builds npu-zoo
docker compose up -d demo                            # web demo of every model, http://<board>:8097
docker compose run --rm smoke                        # all models, one process each; drawings into out/
docker compose run --rm smoke python tests/smoke.py yolov8_l sam2   # just some models
```
`$MODELS` is the model directory, `/mnt/data/npu-models` by default (compose files and `fetch_models.sh`
honour it). `zoo/` (and `demo/`) are bind-mounted, so code changes need only a container restart, not a
rebuild. The smoke test exits 1 if any model failed; look at the images in `out/`, not only the exit code.

## How the pieces fit
* `zoo/__init__.py` `MODELS` is the single registry. Each entry's `task`, `cls` (module.Class under `zoo/`),
  `kw` (constructor args) and `params` (name → (type, default) accepted by `__call__`) drive everything:
  `zoo.get()`, the demo's UI and `/api/run/<model>` form fields, and the smoke test.
* `tests/smoke.py` picks the sample image by model name, else by task (`SAMPLE` dict), and extra call
  args from `EXTRA`. A new model with a new task or odd input needs entries there. The demo's
  `static/index.html` has the same mapping (`DEF_SAMPLE`).
* `zoo/npu.py` `Graph` wraps one `.cix`: one shared NOE context per process, quantize/dequantize per
  tensor descriptor, a job pool for threads.
* Model classes in `zoo/<task>.py` do CPU pre/post-processing around one or more `Graph`s (some also use
  onnxruntime on the CPU, e.g. SAM2 decoder, SigLIP text, YOLO-World's CLIP text tower).
* `demo/app.py` (FastAPI) runs every model through `zoo.worker.ModelWorker`, so it never runs out of NPU memory.

## Ground rules
* **Everything runs in Docker.** There's no NPU runtime on the host Python. The image is `npu-zoo`
  (`docker compose build`): `python:3.12-slim-bookworm` plus the CIX runtime from `vendor/` (libnoe 3.1.2
  wheel and `libaipudrv.so`, fetched by `tools/fetch_runtime.sh`). Never commit `vendor/` binaries,
  model files or anything from the hub: they are under CIX's licence.
* **Containers need** `devices: [/dev/aipu]` and these mounts:
  * `$MODELS/hub/models:/hub:ro`
  * `$MODELS/hf:/hf`
  * `./zoo:/zoo-src/zoo:ro`, with `PYTHONPATH=/zoo-src`
  * See `apps/_template/compose.yml`.
* **Don't run the hub's own scripts or install its requirements** (`$MODELS/hub/...*.py`, torch, the hub
  venv). Read them for reference only. All inference goes through `zoo/npu.py`.
* **Don't edit `zoo/` for one app's needs.** Wrap or subclass in the app. Fix bugs in `zoo/` when they're
  generic, then run `docker compose run --rm smoke python tests/smoke.py <model>`.
* **The NPU may be shared** with other long-running services. The kernel driver queues jobs, so concurrent
  use is fine, but heavy loops (Real-ESRGAN, SigLIP) slow the others down: give them a low duty cycle or a queue.
* **Ports:** the demo uses 8097 and the template 8098. Pick something free (8100+) for new apps.
* **Secrets** (camera passwords, API keys) go in an app-local `.env` with chmod 600, never into code or
  compose files.
* **Sample images** must be public domain or CC0; record each one in `samples/SOURCES.md`.

## Model API (all models)
```python
import zoo                       # registry: zoo.MODELS[name] -> task, class, desc, params
from zoo.common import imread, encode, draw_boxes

m = zoo.get("yolov8_l")          # cached per process; zoo.unload(name) drops it (NPU memory is freed only at exit)
img = imread(path_or_bytes)      # BGR uint8 HxWx3 (OpenCV convention everywhere)
res = m(img, conf=0.3)           # dict; boxes are absolute xyxy pixels in the ORIGINAL image
vis = m.draw(img, res)           # BGR (or BGRA for cut-outs) visualisation
```
Result shapes per task:
* **detect:** `{"detections": [{label, cls, score, box}]}`
* **faces:** `{"faces": [{box, score, kps[5][2]?, emb[512]? (face_id), similarity?, age?, gender?}]}`
* **embed:** `Clip/Siglip.embed_image(img)` and `.embed_text(str)` return L2-normalised vectors; `__call__` does zero-shot labels.
* **segment:**
  * `YoloSeg`: `{"detections", "masks": [HxW uint8]}`
  * `U2Net` / `PPHumanSeg`: `{"alpha": HxW uint8}`
  * `Sam2`: `sam.set_image(img)` once, then `sam.segment(points=[(x, y, 1|0)], box=(x1, y1, x2, y2))` returns `(mask, score)`.
* **pose:** `{"people": [{box, score, keypoints[17][x, y, conf]}]}` (COCO order in `zoo.pose.KPT_NAMES`);
  handpose returns `{"points": [21][x, y]}`.
* **depth:** `{"depth": HxW float32}`, relative inverse depth (bigger = closer).
* **enhance:** `{"image": BGR uint8}`.

Lower-level building blocks:
* `zoo.faces.align` and `ArcFace.embed(face112)` for your own face pipeline
* `zoo.enhance.tiled` for any fixed-size image-to-image model
* `zoo.common.letterbox`, `nms`, `topleft_pad`
* `zoo.npu.Graph(path).run([inputs]) -> [flat float32 outputs]` for a raw `.cix` (quantization handled)

## Starting a new app
1. Copy the template: `cp -r apps/_template apps/<name>`. Rename the service and port in `compose.yml`.
2. Load models at import or startup, not per request. Loading takes 0.1–3 s and NPU memory.
   SAM2's encoder and SigLIP's text tower are the heavy ones.
3. Keep CPU work off the hot path: decode with OpenCV and downscale big photos (the models see ≤ 640 px anyway).
4. Run it with `docker compose -f apps/<name>/compose.yml up -d --build`.
5. Test it with real data (`samples/` or your own) and look at the output images yourself. Several hub
   models fail silently with plausible-looking numbers.
6. Write `apps/<name>/README.md` with what the app does, how to run it, and measured speed.

## Adding a hub model that isn't wrapped yet
1. Fetch the `.cix`: add it to `tools/fetch_models.sh` and run it.
2. Read the hub's `inference_npu.py` and the helpers it calls in `$MODELS/hub/utils/` for
   preprocessing (size, RGB/BGR, mean/std, padding) and output reshapes.
3. Print the tensor descriptors: `Graph(p).ins` / `.outs` show the dtype and quant scale. A uint8 input with
   scale 1 means raw 0..255; scale 255 means 0..1.
4. Write a class following the API above, add it to `zoo.MODELS`, and run the smoke test. Compare against the
   fp32 ONNX on the CPU (onnxruntime is in the image) when the output looks off.

## Gotchas (learned the hard way)
* **YOLO-World:**
  * The large-object head is dead in the `.cix`; the zoo works around it with a second pass at 1/3 scale.
  * Text prompts must use the fp32 CPU CLIP text tower, which the zoo does.
  * Boxes from the 1/3 pass are coarse. When feeding them to SAM2, pad them by about 8 % and add the
    centre point. For boxes covering more than half the image, use the point only (see the template).
* **int8 confidences can saturate:**
  * CenterFace tops out at 0.214.
  * RFB-320 outputs 1.0 for everything.
  * Use them as detected / not detected, and rank with a better model if needed.
* **SCRFD recall:** it misses some clear frontal faces at `conf=0.5` (2 of 7 in `samples/faces.jpg`), and
  very large faces (a face filling a portrait). Lower `conf` or shrink the image when recall matters.
* **Hand pose** needs a tight, roughly square crop of one hand; loose crops give squeezed, wrong landmarks.
* **Upscaling hurts face detectors:** keep `max_up=1.0` (default in `faces.py`) when you resize small images.
* **Process exit noise:** `[UMD ERR] ... ioctl free buffer ... [fail]` and `pure virtual method called` at
  exit are harmless libnoe noise. Scripts can end with `os._exit(0)` to skip them.
* **NPU memory is only freed at process exit (libnoe 3.1.2):**
  * `noe_unload_graph` returns SUCCESS but frees nothing.
  * `noe_deinit_context` does free it, but loading a different graph afterwards segfaults.
  * A process can load roughly 1.5 GB of graphs in its lifetime (e.g. 4× SigLIP), shared with whatever
    else is running on the NPU.
  * **Few fixed models** (the normal app): `zoo.get()` each once at startup and keep them.
  * **Many or changing models:** `zoo.worker.ModelWorker().call(name, img, **params)` runs them in a child
    process that is replaced automatically when a load hits BUF_ALLOC_FAIL. The demo and the smoke test
    work this way.
* **Not on the NPU:** LLMs and VLMs; the hub runs them on the CPU through llama.cpp.

## App ideas that fit these models
* **Photo library:**
  * face clustering and "find this person" (`face_id` embeddings in a vector database)
  * SigLIP re-ranking of CLIP search results
  * depth or cut-out tools
* **Camera** (RTSP):
  * custom YOLO-World alerts ("parcel at the door")
  * person and pose events (fall detection from `yolov8_pose`)
  * low-light clean-up of night snapshots
* **Scans and old photos:** Real-ESRGAN or ESPCN upscaling before OCR, low-light fixes.
