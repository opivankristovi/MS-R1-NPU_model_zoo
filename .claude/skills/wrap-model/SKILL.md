---
name: wrap-model
description: Add a CIX AI Model Hub model that the zoo doesn't wrap yet (fetch the .cix, work out pre/post-processing, write the class, register it, test it). Use when asked to add, wrap or support a new hub model.
argument-hint: "<hub model folder or name>"
---

# Wrap a hub model

The hub's own scripts are reference material only: never run them or install their requirements (torch,
the hub venv). All inference goes through `zoo/npu.py`. The hub clone is at `$MODELS/hub` on the host
(default `/mnt/data/npu-models/hub`) and its `models/` folder is `/hub` inside the containers.

## 1. Fetch
Find the model folder under `$MODELS/hub/models/` (pointer files are present for everything). Add the
`.cix`, plus any CPU-side `.onnx` it needs, to `FILES` in `tools/fetch_models.sh` and run the script.

## 2. Read the reference
Read the model's `inference_npu.py` and every helper it calls in `$MODELS/hub/utils/`
(`image_process.py`, `draw.py`, `tools.py`...). Write down:
* input size, and whether it's a plain resize, a letterbox (where's the padding?) or a crop
* RGB or BGR, the scaling (`/255`? mean and std?), NCHW or NHWC
* how the outputs are reshaped and decoded (anchors, strides, sigmoid already applied or not, NMS)
* how coordinates are mapped back to the original image

## 3. Check the tensors on the NPU
```bash
docker compose run --rm smoke python -c "from zoo.npu import Graph; g = Graph('/hub/<path>.cix'); print(g.ins); print(g.outs); import os; os._exit(0)"
```
Example output: `[<uint8 x196608 scale=255 zp=0>]` / `[<int8 x42 scale=162.62 zp=0>]` (handpose: a 3×256×256
input in 0..1, 42 outputs = 21 × (x, y)).
The descriptors show dtype, shape and quantisation. A uint8 input with scale 1 takes raw 0..255; scale 255
takes 0..1. `Graph.run` quantises inputs and dequantises outputs, and returns flat float32 arrays.

## 4. Write the class
* Put it in the matching `zoo/<task>.py`, load graphs with `load_graph(...)` in `__init__`, and take a lock
  around `run` (see the existing classes).
* `__call__(img, **params)` takes a BGR uint8 image and returns the task's result dict (CLAUDE.md, "Model API").
  Boxes and points are in original-image pixels.
* `draw(img, res)` returns a visualisation.
* Register it in `zoo/__init__.py` `MODELS`: `task`, `cls`, `kw`, a one-line `desc`, and `params` with
  defaults. The demo UI and the smoke test are generated from this entry.
* New task or unusual input: add the sample to `SAMPLE` / `EXTRA` in `tests/smoke.py` and to `DEF_SAMPLE`
  in `demo/static/index.html`. A new sample image must be public domain or CC0; add it to `samples/SOURCES.md`.

## 5. Verify against fp32
Run `/smoke <name>` and look at the output. If anything looks off, compare with the fp32 ONNX on the CPU
(onnxruntime is in the image; the hub folder usually has the `.onnx` under `model/`): same preprocessed
input, compare the raw outputs (cosine similarity, top labels or boxes). int8 differences show up as cosine
0.85–0.99. Much lower means a preprocessing or layout mistake, or a broken `.cix`: YOLO-World's large-object
head is an example of the latter.

## 6. Document
* Add a row to the README model table with the measured warm time and the sample it was measured on.
* Add any defect or workaround to the README "Findings" section, and to the CLAUDE.md gotchas if it
  affects people building apps.
* Update the model count where it appears (README, CLAUDE.md).
