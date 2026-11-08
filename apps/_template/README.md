# Template app: text-prompted cut-out

`POST /cutout` takes `image` (file) and `prompt` (e.g. "truck" or "dog, cat") and returns a transparent PNG of
every matching object. It chains two zoo models: YOLO-World finds boxes for the words, then SAM2 masks each box.

```bash
docker compose -f apps/_template/compose.yml up -d          # port 8098
curl -F image=@samples/truck.jpg -F prompt=truck localhost:8098/cutout -o cut.png
```
Measured: about 2.5 s per new image (YOLO-World 2 passes ~1 s, SAM2 encoder ~1.4 s, decoder ~50 ms per object).
Copy this folder to start a new app (see ../../CLAUDE.md).
