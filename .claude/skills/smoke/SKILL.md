---
name: smoke
description: Run the zoo smoke test for some or all models and check the output images, not just the exit code. Use after any change in zoo/, after adding a model, after rebuilding the image, or when asked whether the models still work.
argument-hint: "[model ...]  (default: all)"
---

# Smoke test with a visual check

Several hub models fail silently: they return plausible numbers and a drawing, with no error. The exit
code only tells you nothing crashed. This skill runs the test and then looks at what came out.

## 1. Run

From the repo root (the image must exist: `docker compose build`):

```bash
docker compose run --rm smoke python tests/smoke.py $ARGUMENTS   # some models
docker compose run --rm smoke                                    # all 26, one process each (~2 min)
```

Run the full set in the background when it's all models. For every model the log prints the input and output
tensor descriptors, `load` and `run` times, the image size and a result summary (truncated at 900 chars).
Drawings go to `out/<model>.jpg|png`. The last line is `FAILED: [...]` or `FAILED: none`.

## 2. Check each output

For every model that ran, hand the output to the `output-checker` subagent, all in one batch:
model name, task, input sample path (`samples/...`, see the mapping below) and output path (`out/...`),
plus the result summary line from the log, **and the expected result from the list below** where there is
one (without it, the checker can only judge plausibility, so a clear regression may come back as SUSPECT
rather than WRONG). When subagents aren't available, open the images yourself and apply the same checks
(`.claude/agents/output-checker.md` lists them).

Which sample each model gets (`SAMPLE` / `EXTRA` in `tests/smoke.py`; the demo's `DEF_SAMPLE` matches):

| model(s) | input |
|---|---|
| detect models, `yolo_world` | `street.jpg` |
| `face_id`, `scrfd`, `centerface`, `rfb_320`, `age_gender` | `faces.jpg` (`face_id` also gets `ref=face_a.jpg`, the bottom-left person) |
| `clip`, `siglip` | `dog.jpg` |
| `yolov8_seg`, `sam2` (`points=1100,700,1`) | `truck.jpg` |
| `u2net` | `portrait.jpg` |
| `yolov8_pose`, `pphumanseg` | `people.jpg` |
| `handpose` | `hand.jpg` |
| `depth_anything`, `midas` | `room.jpg` |
| `real_esrgan`, `espcn`, `vdsr` | `small.png` |
| `agllnet`, `pairlie` | `lowlight.png` |
| `night_dehaze` | `night_haze.png` |

Expected results on these samples (a regression if they change):
* `face_id`: 5 faces at conf 0.5; the bottom-left one has `similarity` ≈ 0.69 and `match: true`, all others ≤ 0.1.
* `yolov8_l` on `street.jpg`: people on the crosswalk and cars, scores ≈ 0.86 for the nearest.
* `yolov8_pose`: 2 large people with full skeletons plus smaller ones in the background.
* `sam2`: the whole pickup truck, cap and wheels included, nothing of the road; score ≈ 0.80, area_fraction ≈ 0.17.
* `handpose`: thumb along the raised thumb (the tip lands about 10 % short of the real tip), four fingers
  curled to the right.

## 3. Report

One line per model: OK, or what looks wrong and why. For timings, compare `run` with the README table and
mention anything more than about 30 % slower. Other NPU users can slow a single run, so repeat that model
once before calling it a regression. Don't update the README timings from one run.
