#!/usr/bin/env bash
# One-time venv install for a FastWAM clone/worktree (runs on the debug pod, CPU only, no torch import).
#   BASE_DIR   : the clone/worktree to install into (default /data/huiwon/fastwam)
#   FREEZE_FROM: optional freeze.txt of an existing venv -> exact same package versions (the `-e file://` line
#                of the source tree is dropped and replaced by `-e $BASE_DIR`)
# NEVER copy a venv between trees (`cp -al`/`cp -r`): bin/* shebangs keep pointing at the SOURCE venv's python
# and training then imports fastwam from the wrong tree. Always build a fresh venv per tree with this script.
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/data/huiwon/.uv-python
export UV_CACHE_DIR=/data/huiwon/.uv-cache
export UV_HTTP_TIMEOUT=600
BASE_DIR="${BASE_DIR:-/data/huiwon/fastwam}"
FREEZE_FROM="${FREEZE_FROM:-}"
cd "$BASE_DIR"
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
if [ -n "$FREEZE_FROM" ] && [ -s "$FREEZE_FROM" ]; then
  grep -v -E '^-e |^fastwam[ =@]' "$FREEZE_FROM" > .venv/requirements.pinned.txt
  uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r .venv/requirements.pinned.txt
  uv pip install --no-deps -e .
else
  uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
  uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e .
fi
uv pip freeze > .venv/freeze.txt
head -1 .venv/bin/torchrun
python -c "import fastwam, sys, os; print('IMPORT_OK', fastwam.__file__, sys.version); assert os.path.realpath(fastwam.__file__).startswith(os.path.realpath('$BASE_DIR')), 'fastwam imported from the wrong tree'"
echo INSTALL_DONE
