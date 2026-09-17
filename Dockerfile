# yue on RunPod serverless (linux/amd64, NVIDIA).
#
# Weights are NOT baked in: HF_HOME points at the network volume, so the first job
# on a fresh volume downloads ~8 GB once and every later cold start reads it from
# disk. torch 2.10's PyPI wheels bundle the CUDA 12.8 runtime, so a slim Python
# base is enough; the host only needs a driver that supports CUDA 12.8.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git ca-certificates libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first so a code change does not re-download torch.
COPY pyproject.toml uv.lock README.md ./
COPY envs/sheetsage2/pyproject.toml envs/sheetsage2/uv.lock ./envs/sheetsage2/
# Both environments in ONE layer, hardlinked from one uv cache that is then deleted.
# SheetSage2 (transcribe / remix from audio) needs its OWN Python 3.11 environment --
# its pins (torch 2.8, transformers 4.45, numpy 1.24) cannot share yue2-infer's -- but
# the nvidia-* CUDA wheels are the same versions in both. Installed as copies in two
# layers they were ~3 GB twice, and exporting + pushing that took the build past an
# hour; hardlinked, the layer tar stores each file once.
# No `--extra cuda`: vLLM adds GBs of image for a backend the default torch path
# (CUDA graphs) does not need. Add it back here when --backend vllm is wanted.
# Same layout as a local checkout, so yuecli.transcribe finds envs/sheetsage2/.venv.
RUN UV_CACHE_DIR=/app/.uv-cache UV_LINK_MODE=hardlink sh -c '\
    uv sync --frozen --no-dev --no-install-project --extra worker \
 && uv sync --frozen --project envs/sheetsage2 --python 3.11 \
 && rm -rf /app/.uv-cache'
COPY src ./src
RUN uv sync --frozen --no-dev --extra worker

CMD ["python", "-m", "yuecli.remote.handler"]
