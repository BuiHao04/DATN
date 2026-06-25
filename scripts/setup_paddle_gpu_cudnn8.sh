#!/usr/bin/env bash
# Legacy helper for the old PaddleOCR 2.x GPU detector.
#
# The project now defaults to PaddleOCR 3.7 / PP-OCRv6 on PaddlePaddle 3.2.1 CPU,
# while VietOCR still uses torch CUDA for recognition. Do not run this script unless
# you intentionally want to roll back to the old paddlepaddle-gpu==2.6.2 stack.
#
# Why this is needed:
#   The GPU Paddle build (paddlepaddle-gpu==2.6.2, cu118) links against cuDNN 8, but
#   this conda env ships cuDNN 9 (required by torch 2.4 / VietOCR). We can't pip-install
#   cuDNN 8 over cuDNN 9 without breaking torch, so we keep a private cuDNN 8 copy in a
#   separate dir. The .so.8 vs .so.9 sonames let both versions coexist in one process
#   (Paddle detect on GPU + VietOCR recognize on GPU).
#
#   Paddle resolves cuDNN through the system loader, so this dir must be on
#   LD_LIBRARY_PATH at process start. The backend injects it automatically for the OCR
#   subprocess (see _build_subprocess_env in app/backend/main.py); src/pipeline/core/
#   ocr_engine.py only enables GPU detect when this dir is present on LD_LIBRARY_PATH.
#
# Usage:
#   bash scripts/setup_paddle_gpu_cudnn8.sh
#   # then (re)start the backend; batch OCR will run detection on the GPU.
set -euo pipefail

if [[ "${ALLOW_LEGACY_PADDLE_GPU_SETUP:-0}" != "1" ]]; then
  echo "This script is legacy and would install paddlepaddle-gpu==2.6.2."
  echo "Current OCR stack expects paddleocr==3.7.0 + paddlepaddle==3.2.1."
  echo "Set ALLOW_LEGACY_PADDLE_GPU_SETUP=1 only if you really want the old stack."
  exit 1
fi

PY="${PYTHON:-$(command -v python)}"
DEST="${OCR_PADDLE_CUDNN8_DIR:-$HOME/.local/lib/paddle_cudnn8}"
CUDNN_VER="${CUDNN8_VERSION:-8.9.6.50}"

echo ">> Installing paddlepaddle-gpu (cu118) if missing..."
"$PY" - <<'PYCHK' || "$PY" -m pip install "paddlepaddle-gpu==2.6.2" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
import paddle, sys
sys.exit(0 if paddle.is_compiled_with_cuda() else 1)
PYCHK

echo ">> Downloading cuDNN 8 (nvidia-cudnn-cu11==$CUDNN_VER) ..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"$PY" -m pip download "nvidia-cudnn-cu11==$CUDNN_VER" -d "$TMP" --no-deps
cd "$TMP"
"$PY" -m wheel unpack ./*.whl -d ext 2>/dev/null || (mkdir -p ext && cd ext && unzip -oq ../*.whl)

echo ">> Installing cuDNN 8 libs into $DEST ..."
mkdir -p "$DEST"
find ext -name 'libcudnn*.so.8' -exec cp -n {} "$DEST"/ \;
# Paddle dlopens the unversioned soname (libcudnn.so), so create the symlinks.
cd "$DEST"
for f in libcudnn*.so.8; do ln -sf "$f" "${f%.8}"; done

echo ">> Done. cuDNN 8 installed at: $DEST"
ls -1 "$DEST"
echo
echo "Restart the backend; OCR batch jobs will now run Paddle detection on the GPU."
