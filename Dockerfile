# NPU model zoo image (npu-zoo). Self-contained: CIX NOE UMD 3.1.2 Python bindings + Arm China driver
# lib from vendor/ (run tools/fetch_runtime.sh first), plus CPU-side helpers
# (onnxruntime for SAM2 decoder / SigLIP text, sentencepiece, HF tokenizer download).
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY vendor/libaipudrv.so vendor/libnoe-*.whl /opt/cix/
RUN pip install --no-cache-dir /opt/cix/libnoe-*.whl numpy opencv-python-headless pillow \
    fastapi uvicorn python-multipart tokenizers onnxruntime sentencepiece huggingface_hub
ENV LD_LIBRARY_PATH=/opt/cix PYTHONUNBUFFERED=1 PYTHONPATH=/zoo-src HF_HOME=/hf ZOO_HUB=/hub
COPY zoo /zoo-src/zoo
COPY demo /zoo-src/demo
WORKDIR /zoo-src
