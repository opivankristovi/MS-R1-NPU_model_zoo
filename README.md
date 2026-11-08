# NPU model zoo (MS-R1 / CIX P1 / Radxa Orion O6, Zhouyi X2 NPU)

26 ready-to-use vision models from the **CIX AI Model Hub 25Q3**, wrapped in one small Python package (`zoo/`) with
the same interface. Each model's pre- and post-processing has been worked out and tested on this board.
There's a web demo to try them, and a template for building apps on them.

## Requirements

* A CIX P1 board (tested on a Radxa Orion O6) with the NPU kernel driver loaded: `cix-npu-driver-dkms`
  6.2.0, which provides `/dev/aipu`. The user-space runtime must match it; `tools/fetch_runtime.sh` fetches
  the matching libnoe 3.1.2 from Radxa's `cix-prebuilt` 26Q2 release.
* Docker with Compose.
* About 7 GB of disk for the models. They go to `$MODELS` (default `/mnt/data/npu-models`).
* A pointer-file clone of the hub, which `tools/fetch_models.sh` fills in (no git-lfs needed):
  `GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/cix/ai_model_hub_25_Q3.git $MODELS/hub`

## Quick start

```bash
tools/fetch_runtime.sh                             # once: CIX NOE runtime (libnoe 3.1.2 + libaipudrv.so) into vendor/
tools/fetch_models.sh                              # once: ~6 GB of .cix/.onnx from ModelScope into $MODELS/hub
docker compose build
docker compose up -d demo                          # http://<board>:8097
docker compose run --rm smoke                      # run every model on samples/, drawings into out/
```

```python
import zoo
from zoo.common import imread
m = zoo.get("yolo_world")                          # loads once per process
res = m(imread("x.jpg"), classes="red mug, laptop")  # -> {"detections": [{label, score, box[x1,y1,x2,y2]}]}
cv2.imwrite("vis.jpg", m.draw(imread("x.jpg"), res))
```

## Models

Times are measured here (2026-11-08). They are warm runs of the whole call (CPU pre-processing, NPU,
CPU post-processing) on the sample image in brackets.

| name | what | time |
|---|---|---|
| **Detection** | | |
| `yolov8_l` | YOLOv8-L, 80 COCO classes | 44 ms |
| `yolov12_m` | YOLOv12-M, 80 COCO classes | 79 ms |
| `yolox_m` / `yolox_l` | YOLOX (same model as Frigate) | 28 / 37 ms |
| `yolo_world` | **open vocabulary**: name up to 80 things ("parcel, ladder, cat on a table") | 0.98 s (2 passes) |
| **Faces** | | |
| `face_id` | SCRFD-10G detection + ArcFace-R50 512-d identity embedding, optional reference face | 72 ms (5 faces) |
| `scrfd` / `centerface` / `rfb_320` | face detectors (5 landmarks / 5 landmarks / none) | 28 / 16 / 19 ms |
| `age_gender` | SCRFD + GoogLeNet age bucket and gender (rough) | 50 ms (5 faces) |
| **Zero-shot / embeddings** | | |
| `clip` | CLIP ViT-B/32: image and text, both on the NPU | 17 ms per image or text |
| `siglip` | SigLIP so400m-384: better embeddings, image on NPU, text on CPU | 1.07 s image, 0.45 s per text |
| **Segmentation** | | |
| `yolov8_seg` | YOLOv8-L instance masks (COCO) | 0.5 s (1800×1350, CPU mask post-processing) |
| `sam2` | SAM 2.1 tiny, segment what you click / box. Encoder on NPU, decoder on CPU | ~1.4 s per new image, ~60 ms per extra prompt |
| `u2net` | human cut-out, alpha matte → transparent PNG | 47 ms |
| `pphumanseg` | person mask, very fast | 19 ms |
| **Pose** | | |
| `yolov8_pose` | people with 17 keypoints | 48 ms |
| `handpose` | 21 hand landmarks; the input must be a crop of one hand | 19 ms |
| **Depth** | | |
| `depth_anything` | Depth-Anything-V2, relative depth | 289 ms |
| `midas` | MiDaS v2 small, relative depth | 41 ms |
| **Enhancement** | | |
| `real_esrgan` | 4× upscaling and artefact removal (224 px tiles) | 12.6 s (512×340 → 2048×1360) |
| `espcn` | fast 3× upscaling (luma) | 47 ms (512×340) |
| `vdsr` | 2–4× luma refinement | 0.10 s (512×340, 2×) |
| `agllnet` / `pairlie` | low-light enhancement | 100 / 79 ms (1080×588) |
| `night_dehaze` | night haze and glow removal | 1.0 s |

## Findings (what had to be fixed versus the hub's scripts)

* **YOLO-World's large-object head is broken in the hub `.cix`.**
  * The stride-32 scores are about 0.001 for everything, so objects bigger than about 1/3 of the frame
    are never found. A full-frame truck scored 0.01 at full and half scale and 0.62–0.82 at 1/3 scale.
  * `zoo.detect.YoloWorld` therefore runs a second pass with the image shrunk to 1/3 and merges the results
    (`multiscale=True`). Boxes from that pass are coarse, at about ±3 % of the image size.
* **YOLO-World needs fp32 CLIP text embeddings.**
  * With the int8 NPU text tower (cosine 0.95 to fp32), "person" scored 0.03 instead of 0.82.
  * Class prompts are therefore embedded on the CPU with `clip_text_model_vitb32.onnx` (54 ms per prompt, cached).
* **CenterFace's int8 heatmap saturates at 0.214.** Scores carry no information; treat it as a yes/no
  detector at threshold 0.2.
* **SCRFD and CenterFace miss very large faces** when a small image (e.g. a 250 px portrait) is upscaled to
  640. Upscaling is now capped at 1×.
* **NPU memory leaks when graphs are unloaded (libnoe 3.1.2).** It is only freed at process exit.
  The demo runs its models in a `zoo.worker.ModelWorker` child process, which is replaced when memory
  runs out. The smoke test uses one process per model.
* **Fixed-size inputs.**
  * Super-resolution models take 224–256 px inputs, so they run on overlapping, blended tiles.
  * Enhancement models take 256–512 px inputs. They run on a downscaled copy, and the original's
    high-frequency detail is added back, so the output keeps the full resolution.
* **Accuracy versus fp32:**
  * SigLIP image tower: cosine 0.82–0.88, with the same top label on 4 of 4 test images. fp32 on the CPU
    takes 5.6 s per image.
  * CLIP: cosine 0.85–0.97.
* **Face matching:** two different photos of the same person (`face_a.jpg` and the bottom-left face in
  `faces.jpg`) scored 0.69; the other people scored at most 0.09. The threshold is 0.35.
* **SCRFD misses some clear faces.** In `faces.jpg` it finds 5 of 7 frontal faces at the default
  `conf=0.5`: one scores only 0.20 and one isn't detected at all. Lower `conf` (about 0.15) when recall matters.
* **Hand pose needs a tight, square crop of one hand.** A loose or tall crop gives landmarks squeezed to
  one side, with no error.
* **Hub models left out** (no clear use here): openpose and hrnet (`yolov8_pose` is better and simpler),
  ImageNet classifiers, lane and BEV road models, the Chinese CLIP, `fsre_depth`, `facenet`, and
  stable diffusion (worth a separate try).

## Layout

| path | what |
|---|---|
| `zoo/` | the package; `zoo/__init__.py` has the `MODELS` registry. Modules: `detect`, `faces`, `embed`, `segment`, `pose`, `depth`, `enhance`, `common`, `npu`, `worker` |
| `demo/` | FastAPI and one-page UI for trying every model (port 8097) |
| `apps/_template/` | a starter app (text-prompted cut-out: YOLO-World → SAM2), port 8098 |
| `tests/smoke.py` | runs all models, prints I/O tensors, timings and results |
| `tools/fetch_models.sh` | downloads exactly the files the zoo needs |
| `samples/` | test images (public domain / CC0, see `samples/SOURCES.md`) |
| `tools/fetch_runtime.sh` | downloads the CIX NOE runtime into `vendor/` (not committed) |
| `CLAUDE.md` | how to build apps here (for Claude Code and for humans) |

Models stay in `$MODELS/hub/models` (mounted read-only at `/hub`). The CLIP tokenizer is cached in `$MODELS/hf`.

## Licence

The code in this repository is under the [Apache License 2.0](LICENSE). It contains no models and no CIX
binaries:

* The models are downloaded by `tools/fetch_models.sh` from the
  [CIX AI Model Hub](https://www.modelscope.cn/models/cix/ai_model_hub_25_Q3). They are under CIX's terms
  (see the hub's `LICENSE` and `EULA.md`) and the original licences of each model.
* The NOE runtime is downloaded by `tools/fetch_runtime.sh` from
  [radxa-pkg/cix-prebuilt](https://github.com/radxa-pkg/cix-prebuilt/releases/tag/26Q2-2607), under CIX's terms.
* The sample images are public domain or CC0; `samples/SOURCES.md` lists where each one comes from.
