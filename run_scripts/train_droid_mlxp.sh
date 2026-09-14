#!/usr/bin/env bash
# =====================================================================================================
# FastWAM x DROID on Naver MLXP — launcher (single node, $NUM_GPUS GPUs), idempotent preflight + auto-resume
# -----------------------------------------------------------------------------------------------------
#   task     : configs/task/droid_fastwam_5b_b128_200k.yaml  (data=droid_wan22_5b, model=fastwam_droid)
#   mode     : uncond FastWAM = "first-frame" Fast-WAM (fastwam.runtime.create_fastwam), no test-time imagination
#   batch    : PER_DEV x NUM_GPUS x GA = 32 x 4 x 1 = 128 (OOM_FALLBACK=1: +grad-ckpt; =2: pd16 x GA2 + grad-ckpt)
#   ckpt     : $MODEL_OUTPUT_DIR/<run>/checkpoints/{weights/step_NNNNNN.pt, state/step_NNNNNN/} every 1000, keep 5
#   resume   : automatic — newest state/step_* dir that contains trainer_state.json -> `resume=<dir>`
#              (accelerate.load_state: DiT + ZeRO-1 optimizer + LR sched; trainer_state.json: step/epoch/
#              batch_in_epoch -> ResumableEpochSampler offset). Nothing to pass by hand.
#   one-time : (a) ActionDiT backbone .pt (GPU, ~min)  (b) dataset_stats.json (CPU parquet pass)
#              (c) umT5 text-embedding cache for every tasks.jsonl row (torchrun, all GPUs)
#              — each runs only when its output is missing.
#   logs     : /data/huiwon/logs/gr00t-vdm/<TAG>-<ts>.{out,err} (or the caller's $FASTWAM_LOG_PREFIX)
# =====================================================================================================
set -uo pipefail

TAG="${TAG:-fastwam-droid-wan22-5b-b128-200k-4gpu}"
RUN_NAME="${RUN_NAME:-fastwam_droid_wan22_5b_b128_200k_4gpu}"
TASK="${TASK:-droid_fastwam_5b_b128_200k}"
BASE_DIR="${BASE_DIR:-/data/huiwon/fastwam}"
LOG_DIR="${LOG_DIR:-/data/huiwon/logs/gr00t-vdm}"
PREP_DIR="${PREP_DIR:-/data/huiwon/data/droid_fastwam}"
CKPT_BASE="${CKPT_BASE:-/data/huiwon/checkpoints/fastwam_base}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/data/rlwrld-unified-checkpoints/huiwon/fastwam}"

mkdir -p "$LOG_DIR"
if [ -z "${FASTWAM_LOG_PREFIX:-}" ]; then
  L="$LOG_DIR/$TAG-$(date +%Y%m%d_%H%M%S)"
  exec > >(tee -a "$L.out") 2> >(tee -a "$L.err" >&2)
else
  L="$FASTWAM_LOG_PREFIX"
fi
echo "[$TAG] launcher START $(date -u +%FT%TZ) host=$(hostname) log=$L.{out,err}"
fatal() { echo "[$TAG] FATAL: $*"; echo "[$TAG] rc=1 $(date -u +%FT%TZ)"; exit 1; }
# wandb: decided BEFORE `set -x` so the key value never reaches the log (xtrace would echo the comparison)
export WANDB_ENTITY="${WANDB_ENTITY:-huiwoen0516}" WANDB_PROJECT=fastwam WANDB_RUN_ID="$RUN_NAME" WANDB_RESUME=allow
WANDB_MODE_OVERRIDE=()
if [ -z "${WANDB_API_KEY:-}" ] || [[ "${WANDB_API_KEY:-}" == REPLACE_WITH* ]]; then
  echo "[$TAG] WARN: no usable WANDB_API_KEY -> wandb.mode=offline"
  export WANDB_MODE=offline; WANDB_MODE_OVERRIDE=("wandb.mode=offline")
fi
set -x

# ── batch plate ─────────────────────────────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
PER_DEV="${PER_DEV:-32}"; GA="${GA:-1}"; GC="${GC:-false}"
case "${OOM_FALLBACK:-0}" in
  0) ;;
  1) GC=true ;;                    # same batch, activation checkpointing in every MoT block
  2) PER_DEV=16; GA=2; GC=true ;;  # half micro-batch, 2x accumulation (see GA note below)
  *) fatal "OOM_FALLBACK must be 0/1/2" ;;
esac
MAX_STEPS="${MAX_STEPS:-200000}"; SAVE_EVERY="${SAVE_EVERY:-1000}"; SAVE_LIMIT="${SAVE_LIMIT:-5}"; NW="${NW:-12}"
EXPECT_EFF="${EXPECT_EFF:-128}"
EFF=$(( PER_DEV * NUM_GPUS * GA ))
[ "$EFF" -eq "$EXPECT_EFF" ] || fatal "effective batch $EFF != $EXPECT_EFF (pd$PER_DEV x $NUM_GPUS gpu x GA$GA)"
if [ "$GA" -gt 1 ] && [ "${ALLOW_GA_GT1:-0}" != 1 ]; then
  # FastWAM pins deepspeed==0.18.7; our DreamZero/WAM notes: 0.17.x drops micro-batches under ZeRO-2 (#7718) and
  # 0.18.3-0.19.5 carry the #8224 regression. Verify GA>1 correctness (or upgrade to >=0.19.6) before allowing it.
  fatal "GA=$GA > 1 refused with the pinned deepspeed (set ALLOW_GA_GT1=1 after verifying / upgrading deepspeed >= 0.19.6)"
fi

# ── environment ─────────────────────────────────────────────────────────────────────────────────────
export DIFFSYNTH_MODEL_BASE_PATH="$CKPT_BASE" DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export FASTWAM_ACTION_DIT_PT="$CKPT_BASE/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
export DROID_DATA_ROOT="${DROID_DATA_ROOT:-/data/shared_dataset/DreamZero-DROID-Data}"
export FASTWAM_DROID_TEXT_CACHE="$PREP_DIR/text_embeds_cache"
export FASTWAM_VIDEO_BACKEND="${FASTWAM_VIDEO_BACKEND:-torchcodec}"
STATS_JSON="$PREP_DIR/dataset_stats.json"
export HF_HOME=/data/huiwon/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/data/huiwon/.cache/hf_datasets_fastwam
export TMPDIR=/data/huiwon/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=/data/huiwon/.cache/torchinductor_fastwam
mkdir -p "$PREP_DIR" "$HF_DATASETS_CACHE" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

# ── preflight gates ─────────────────────────────────────────────────────────────────────────────────
cd "$BASE_DIR" || fatal "no clone at $BASE_DIR"
echo "[$TAG] repo branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit) HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
PY="$BASE_DIR/.venv/bin/python"
test -x "$PY" || fatal "no venv at $BASE_DIR/.venv (run run_scripts/install_venv.sh on the debug pod)"
# shellcheck disable=SC1091
source "$BASE_DIR/.venv/bin/activate"
case "$(python -V 2>&1)" in *" 3.10."*) ;; *) fatal "venv is not python 3.10: $(python -V 2>&1)";; esac
python -c "import fastwam, torch, accelerate, deepspeed, transformers, hydra; print('imports OK torch', torch.__version__, 'deepspeed', deepspeed.__version__, 'accelerate', accelerate.__version__, 'transformers', transformers.__version__)" || fatal "fastwam env incomplete"
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')" || fatal "torchcodec unusable (ffmpeg shared libs missing?)"
python -m py_compile scripts/train.py scripts/precompute_text_embeds.py scripts/preprocess_action_dit_backbone.py \
  scripts/compute_dataset_stats.py scripts/compose_check.py scripts/check_wan22_weights.py \
  src/fastwam/trainer.py src/fastwam/runtime.py src/fastwam/datasets/lerobot/robot_video_dataset.py \
  src/fastwam/datasets/lerobot/droid_dataset.py src/fastwam/datasets/lerobot/transforms/droid.py || fatal "python syntax"
for f in configs/task/$TASK.yaml configs/data/droid_wan22_5b.yaml configs/model/fastwam_droid.yaml \
         scripts/accelerate_configs/accelerate_zero1_ds.yaml scripts/ds_configs/ds_zero1_config.json; do
  test -s "$f" || fatal "missing config $f"
done
test -s "$DROID_DATA_ROOT/meta/info.json" || fatal "dataset root unreadable: $DROID_DATA_ROOT"
test -d "$DROID_DATA_ROOT/videos" || fatal "dataset videos missing"
test -d "$CKPT_BASE/Wan-AI/Wan2.2-TI2V-5B" || fatal "weight base layout missing under $CKPT_BASE"
python scripts/check_wan22_weights.py --base "$CKPT_BASE" || fatal "Wan2.2 weight files do not match FastWAM's registry"
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L | wc -l); [ "$NGPU" -ge "$NUM_GPUS" ] || fatal "only $NGPU GPUs visible, need $NUM_GPUS"
fi

# ── one-time preprocessing (only when the output is missing) ────────────────────────────────────────
if [ ! -s "$FASTWAM_ACTION_DIT_PT" ]; then
  echo "[$TAG] prep(a): ActionDiT backbone (linear-interp Wan2.2 DiT, alpha-scaled, 1024 hdim) -> $FASTWAM_ACTION_DIT_PT"
  python scripts/preprocess_action_dit_backbone.py --model-config configs/model/fastwam_droid.yaml \
    --output "$FASTWAM_ACTION_DIT_PT.tmp" --device cuda --dtype bfloat16 \
    && mv -f "$FASTWAM_ACTION_DIT_PT.tmp" "$FASTWAM_ACTION_DIT_PT" || fatal "ActionDiT backbone preprocessing"
else
  echo "[$TAG] prep(a): ActionDiT backbone present ($(du -h "$FASTWAM_ACTION_DIT_PT" | cut -f1))"
fi
if [ ! -s "$STATS_JSON" ]; then
  echo "[$TAG] prep(b): dataset_stats.json (CPU pass over all parquet episodes, post-transform q01/q99) -> $STATS_JSON"
  env -u FASTWAM_DROID_STATS python scripts/compute_dataset_stats.py task="$TASK" +stats_output_dir="$PREP_DIR" || fatal "dataset stats"
else
  echo "[$TAG] prep(b): dataset stats present: $STATS_JSON"
fi
export FASTWAM_DROID_STATS="$STATS_JSON"
TXT_DONE="$FASTWAM_DROID_TEXT_CACHE/.precompute_done"
if [ ! -f "$TXT_DONE" ]; then
  echo "[$TAG] prep(c): umT5-XXL text-embedding cache for every meta/tasks.jsonl prompt -> $FASTWAM_DROID_TEXT_CACHE"
  mkdir -p "$FASTWAM_DROID_TEXT_CACHE"
  torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/precompute_text_embeds.py task="$TASK" +overwrite=false \
    && touch "$TXT_DONE" || fatal "text embedding precompute"
else
  echo "[$TAG] prep(c): text cache present ($(ls "$FASTWAM_DROID_TEXT_CACHE" | wc -l) files)"
fi

# ── compose gate (hydra only; proves the resolved config carries the plate above) ───────────────────
python scripts/compose_check.py --task "$TASK" --world "$NUM_GPUS" --expect-effective "$EXPECT_EFF" --check-paths --check-meta -- \
  batch_size="$PER_DEV" gradient_accumulation_steps="$GA" model.mot_checkpoint_mixed_attn="$GC" \
  max_steps="$MAX_STEPS" save_every="$SAVE_EVERY" save_total_limit="$SAVE_LIMIT" num_workers="$NW" || fatal "compose gate"

# ── auto-resume ─────────────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR="$MODEL_OUTPUT_DIR/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
RESUME_DIR=""; LATEST=-1
for d in "$OUTPUT_DIR"/checkpoints/state/step_*/; do
  d="${d%/}"
  [ -f "$d/trainer_state.json" ] || { echo "[$TAG] resume: skipping incomplete $d"; continue; }
  n="$(basename "$d")"; s=$((10#${n#step_}))
  if [ "$s" -gt "$LATEST" ]; then LATEST=$s; RESUME_DIR="$d"; fi
done
shopt -u nullglob
RESUME_ARGS=()
if [ -n "$RESUME_DIR" ]; then
  GS=$(python -c "import json,sys; print(int(json.load(open(sys.argv[1]))['global_step']))" "$RESUME_DIR/trainer_state.json") || fatal "unreadable $RESUME_DIR/trainer_state.json"
  if [ "$GS" -ge "$MAX_STEPS" ]; then
    echo "[$TAG] resume: latest checkpoint step $GS >= max_steps $MAX_STEPS -> run already finished, nothing to do"
    echo "[$TAG] rc=0 $(date -u +%FT%TZ)"; exit 0
  fi
  RESUME_ARGS=("resume=$RESUME_DIR")
fi

echo "[$TAG] BANNER eff_batch=$EFF = ${NUM_GPUS}gpu x pd${PER_DEV} x GA${GA} | grad_ckpt=$GC | max_steps=$MAX_STEPS save_every=$SAVE_EVERY keep=$SAVE_LIMIT nw=$NW | resume=${RESUME_DIR:-fresh(step 0)} | out=$OUTPUT_DIR"

# ── train (mirrors scripts/train_zero1.sh, but with a FIXED output_dir so resume is possible) ────────
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "$NUM_GPUS" --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${MASTER_PORT:-29500}" \
  scripts/train.py \
  task="$TASK" \
  output_dir="$OUTPUT_DIR" \
  batch_size="$PER_DEV" gradient_accumulation_steps="$GA" model.mot_checkpoint_mixed_attn="$GC" \
  max_steps="$MAX_STEPS" save_every="$SAVE_EVERY" save_total_limit="$SAVE_LIMIT" num_workers="$NW" \
  wandb.name="$RUN_NAME" \
  ${WANDB_MODE_OVERRIDE[@]+"${WANDB_MODE_OVERRIDE[@]}"} \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  "$@"
rc=$?
echo "[$TAG] rc=$rc $(date -u +%FT%TZ)"
exit $rc
