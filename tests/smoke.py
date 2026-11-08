"""Run every zoo model on its sample image; print I/O tensors, timings and a result summary; save drawings.

    docker compose run --rm smoke                       # all models
    docker compose run --rm smoke python tests/smoke.py yolov8_l sam2   # some
Outputs go to ./out/<model>.{jpg,png}. Exit code 1 if any model failed.
"""
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

import zoo
from zoo.common import imread

S = os.environ.get("ZOO_SAMPLES", "/samples")
OUT = os.environ.get("ZOO_OUT", "/out")
SAMPLE = {"detect": "street.jpg", "faces": "faces.jpg", "embed": "dog.jpg", "segment": "truck.jpg",
          "pose": "people.jpg", "depth": "room.jpg", "enhance": "small.png", "agllnet": "lowlight.png",
          "pairlie": "lowlight.png", "night_dehaze": "night_haze.png", "handpose": "hand.jpg",
          "u2net": "portrait.jpg", "pphumanseg": "people.jpg"}
EXTRA = {"face_id": {"ref": "face_a.jpg"}, "sam2": {"points": "1100,700,1"}}


def summary(res):
    out = {}
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            out[k] = f"array{list(v.shape)} {v.dtype} [{v.min():.3g}..{v.max():.3g}]"
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [{kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in d.items()
                       if kk not in ("emb", "keypoints", "kps")} for d in v[:6]] + ([f"... {len(v)} total"] if len(v) > 6 else [])
        elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            out[k] = f"{len(v)} arrays"
        else:
            out[k] = v
    return out


def main(names):
    os.makedirs(OUT, exist_ok=True)
    failed = []
    for name in names:
        spec = zoo.MODELS[name]
        print(f"\n=== {name}: {spec['desc']}")
        try:
            t0 = time.perf_counter()
            m = zoo.get(name)
            load = time.perf_counter() - t0
            for g in zoo._graphs(m):
                print("   in ", g.ins, "\n   out", g.outs)
            img = imread(f"{S}/{SAMPLE.get(name, SAMPLE[spec['task']])}")
            kw = {}
            for k, v in EXTRA.get(name, {}).items():
                kw[k] = imread(f"{S}/{v}") if spec["params"].get(k, ("",))[0] == "image" else v
            m(img, **kw)  # warm-up (lazy CPU parts, caches)
            t0 = time.perf_counter()
            res = m(img, **kw)
            dt = time.perf_counter() - t0
            vis = m.draw(img, res)
            ext = ".png" if vis.ndim == 3 and vis.shape[2] == 4 or spec["task"] == "enhance" else ".jpg"
            cv2.imwrite(f"{OUT}/{name}{ext}", vis)
            print(f"   load {load:.1f}s  run {dt*1000:.0f} ms  image {img.shape[1]}x{img.shape[0]} -> {vis.shape[1]}x{vis.shape[0]}")
            print("  ", json.dumps(summary(res), default=str)[:900])
        except Exception:
            traceback.print_exc()
            failed.append(name)
    print("\nFAILED:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    names = sys.argv[1:] or list(zoo.MODELS)
    if len(names) == 1:
        rc = main(names)
        sys.stdout.flush()
        os._exit(rc)  # skip libnoe's noisy teardown
    # one process per model: libnoe only returns NPU memory at process exit
    import subprocess
    failed = [n for n in names if subprocess.run([sys.executable, __file__, n],
                                                  stderr=subprocess.STDOUT).returncode != 0]
    print("\n#### ALL MODELS - FAILED:", failed or "none")
    sys.exit(1 if failed else 0)
