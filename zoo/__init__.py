"""CIX P1 NPU model zoo.  `from zoo import get; m = get("yolov8_l"); res = m(img); vis = m.draw(img, res)`

MODELS is the single registry used by the demo app, the smoke test and new applications.
Each entry: task, class path, constructor kwargs, a one-line description, and the UI parameters
(name -> (type, default)) its __call__ accepts besides the image.
"""
import importlib
import threading

MODELS = {
    # ---------------------------------------------------------------- detection
    "yolov8_l":     dict(task="detect", cls="detect.Yolov8", kw={"variant": "yolov8_l"},
                         desc="YOLOv8-L, 80 COCO classes", params={"conf": ("float", 0.3), "iou": ("float", 0.45)}),
    "yolov12_m":    dict(task="detect", cls="detect.Yolov8", kw={"variant": "yolov12_m"},
                         desc="YOLOv12-M, 80 COCO classes", params={"conf": ("float", 0.3), "iou": ("float", 0.45)}),
    "yolox_m":      dict(task="detect", cls="detect.Yolox", kw={"variant": "yolox_m"},
                         desc="YOLOX-M, 80 COCO classes (used by Frigate)", params={"conf": ("float", 0.35), "iou": ("float", 0.45)}),
    "yolox_l":      dict(task="detect", cls="detect.Yolox", kw={"variant": "yolox_l"},
                         desc="YOLOX-L, 80 COCO classes", params={"conf": ("float", 0.35), "iou": ("float", 0.45)}),
    "yolo_world":   dict(task="detect", cls="detect.YoloWorld",
                         desc="YOLOv8-L-World v2: open-vocabulary detection, name the objects (comma-separated)",
                         params={"classes": ("str", "person, dog, bicycle, backpack"), "conf": ("float", 0.25), "iou": ("float", 0.45)}),
    # ---------------------------------------------------------------- faces
    "face_id":      dict(task="faces", cls="faces.FaceRecognizer",
                         desc="SCRFD-10G + ArcFace-R50: faces, landmarks, 512-d identity embeddings; optional reference face",
                         params={"conf": ("float", 0.5), "ref": ("image", None)}),
    "scrfd":        dict(task="faces", cls="faces.Scrfd", desc="SCRFD-10G face detector with 5 landmarks",
                         params={"conf": ("float", 0.5)}),
    "centerface":   dict(task="faces", cls="faces.CenterFace", desc="CenterFace detector with 5 landmarks",
                         params={"conf": ("float", 0.2)}),
    "rfb_320":      dict(task="faces", cls="faces.Rfb320", desc="Ultra-light RFB-320 face detector (tiny, fast)",
                         params={"conf": ("float", 0.7)}),
    "age_gender":   dict(task="faces", cls="faces.AgeGender", desc="SCRFD faces + GoogLeNet age bucket and gender",
                         params={"conf": ("float", 0.5)}),
    # ---------------------------------------------------------------- embeddings / zero-shot
    "clip":         dict(task="embed", cls="embed.Clip", desc="CLIP ViT-B/32 zero-shot labels (both towers on NPU)",
                         params={"labels": ("str", "a photo of a dog, a photo of a cat, a photo of a person, a photo of a car")}),
    "siglip":       dict(task="embed", cls="embed.Siglip", desc="SigLIP so400m zero-shot labels (image NPU, text CPU)",
                         params={"labels": ("str", "a photo of a dog, a photo of a cat, a photo of a person, a photo of a car")}),
    # ---------------------------------------------------------------- segmentation
    "yolov8_seg":   dict(task="segment", cls="segment.YoloSeg", desc="YOLOv8-L instance segmentation (COCO)",
                         params={"conf": ("float", 0.3), "iou": ("float", 0.45)}),
    "sam2":         dict(task="segment", cls="segment.Sam2",
                         desc="SAM 2.1 tiny: click points (x,y,label;...) or a box -> mask. Encoder NPU, decoder CPU",
                         params={"points": ("str", ""), "box": ("str", "")}),
    "u2net":        dict(task="segment", cls="segment.U2Net", desc="U2-Net human cut-out (transparent PNG)", params={}),
    "pphumanseg":   dict(task="segment", cls="segment.PPHumanSeg", desc="PP-HumanSeg person mask (very fast)",
                         params={"thresh": ("float", 0.5)}),
    # ---------------------------------------------------------------- pose
    "yolov8_pose":  dict(task="pose", cls="pose.YoloPose", desc="YOLOv8s-pose: people + 17 body keypoints",
                         params={"conf": ("float", 0.4)}),
    "handpose":     dict(task="pose", cls="pose.HandPose", desc="21 hand landmarks (input = crop of one hand)", params={}),
    # ---------------------------------------------------------------- depth
    "depth_anything": dict(task="depth", cls="depth.DepthAnything", desc="Depth-Anything-V2 relative depth (518 px)", params={}),
    "midas":        dict(task="depth", cls="depth.Midas", desc="MiDaS v2 small relative depth (256 px, fast)", params={}),
    # ---------------------------------------------------------------- enhancement
    "real_esrgan":  dict(task="enhance", cls="enhance.RealEsrgan", desc="Real-ESRGAN 4x upscaling + artefact removal (tiled)", params={}),
    "espcn":        dict(task="enhance", cls="enhance.Espcn", desc="ESPCN fast 3x upscaling (luma, tiled)", params={}),
    "vdsr":         dict(task="enhance", cls="enhance.Vdsr", desc="VDSR 2-4x upscaling (luma refinement, tiled)",
                         params={"scale": ("int", 2)}),
    "agllnet":      dict(task="enhance", cls="enhance.Agllnet", desc="AGLLNet low-light enhancement", params={}),
    "pairlie":      dict(task="enhance", cls="enhance.Pairlie", desc="PairLIE low-light enhancement",
                         params={"exposure": ("float", 0.5)}),
    "night_dehaze": dict(task="enhance", cls="enhance.NightDehaze", desc="Night-time dehazing / glow removal", params={}),
}

_loaded, _lock = {}, threading.Lock()


def get(name):
    """Load (once per process) and return the model instance for a registry name.

    NPU memory is only returned when the process exits (libnoe 3.1.2 leaks unloaded graphs), so a
    process can hold roughly 1.5 GB of graphs in its lifetime. Apps that use a few models: just call get().
    Apps that switch between many models: use zoo.worker.ModelWorker (models in a child process that is
    replaced when memory runs out)."""
    with _lock:
        if name not in _loaded:
            spec = MODELS[name]
            mod, cls = spec["cls"].rsplit(".", 1)
            _loaded[name] = getattr(importlib.import_module(f"zoo.{mod}"), cls)(**spec.get("kw", {}))
        return _loaded[name]


def loaded():
    return list(_loaded)


def _graphs(obj, seen=None):
    from .npu import Graph
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    for v in vars(obj).values() if hasattr(obj, "__dict__") else []:
        if isinstance(v, Graph):
            yield v
        elif hasattr(v, "__dict__") and type(v).__module__.startswith("zoo"):
            yield from _graphs(v, seen)


def unload(name):
    """Drop a loaded model and close its graphs. NOTE: this does NOT give NPU memory back on
    libnoe 3.1.2 (only process exit does) - it only stops the model being used."""
    with _lock:
        m = _loaded.pop(name, None)
    for g in _graphs(m) if m is not None else []:
        g.close()
