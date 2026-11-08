---
name: new-app
description: Create a new application under apps/ on top of the zoo models (a small FastAPI Docker service copied from apps/_template). Use when asked to build an app, service, tool or demo that uses the NPU models.
argument-hint: "<app-name> <what it should do>"
---

# New zoo app

Read `README.md` (model table, timings, known defects) and the gotchas in `CLAUDE.md` before picking models.
`apps/_template/` is a working example: text-prompted cut-out, YOLO-World → SAM2.

## 1. Plan
* Pick the fewest models that do the job, and check the README for their speed and known defects.
  Say up front what the per-request time will be (sum of the model times plus CPU work).
* Few fixed models (the normal case): `zoo.get(name)` each once at import time.
  Many or user-chosen models: use `zoo.worker.ModelWorker` (NPU memory is only freed at process exit).
* Heavy models in a loop (Real-ESRGAN, SigLIP, SAM2 encoder) slow down everything else on the NPU:
  queue requests or limit the duty cycle.

## 2. Scaffold
```bash
cp -r apps/_template apps/<name>
```
* `compose.yml`: rename the service, pick a free host port ≥ 8100. Check what's in use with
  `docker ps --format '{{.Ports}}'` and `ss -ltn`, and in `CLAUDE.local.md` if it exists.
  Keep the device, the volumes and the environment exactly as they are.
* Secrets (camera URLs, API keys) go in `apps/<name>/.env` (chmod 600, already in `.gitignore`), loaded with
  `env_file:`. Never put them in code or in `compose.yml`.
* Replace `app.py`. Keep the conventions from the template's docstring: BGR uint8 arrays,
  `zoo.common.imread` / `encode`, downscale big photos before inference (the models see ≤ 640 px).
* Extra Python packages: add a `Dockerfile` in the app folder (`FROM npu-zoo` + `pip install`) and
  use `build: .` in `compose.yml` instead of `image: npu-zoo`.

## 3. Run and test with real data
```bash
docker compose -f apps/<name>/compose.yml up -d --build
docker compose -f apps/<name>/compose.yml logs -f --tail 50
```
* Call it with the images in `samples/` (see `samples/SOURCES.md` for what each shows) or the user's data.
* Save the outputs and **look at them**. When the app draws or returns images, hand them with the input to
  the `output-checker` subagent. A plausible-looking number is not proof the model worked.
* Measure: warm request time (second call onwards) and the first-call time.

## 4. Document
Write `apps/<name>/README.md`: what it does, the models it uses, how to run and call it (a `curl` line),
and the measured times. Add a line for the app to the README's layout table if it's meant to stay.
