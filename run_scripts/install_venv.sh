#!/usr/bin/env bash
# One-time venv install for /data/huiwon/fastwam (runs on the debug pod, CPU only, no torch import).
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/data/huiwon/.uv-python
export UV_CACHE_DIR=/data/huiwon/.uv-cache
export UV_HTTP_TIMEOUT=600
cd /data/huiwon/fastwam
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e .
uv pip freeze > .venv/freeze.txt
python -c "import fastwam, sys; print('IMPORT_OK', fastwam.__file__, sys.version)"
echo INSTALL_DONE
