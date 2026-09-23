#!/usr/bin/env bash
# =====================================================================================================
# FastWAM x openarm_wam_v1 (robot + human, rh11) on Naver MLXP — launcher (single node, $NUM_GPUS GPUs),
# idempotent preflight + one-time prep + auto-resume. Derived from run_scripts/train_droid_mlxp.sh.
# -----------------------------------------------------------------------------------------------------
#   task     : configs/task/openarm_fastwam_5b_b64_50k.yaml  (data=openarm_wan22_5b, model=fastwam_droid)
#   mode     : uncond FastWAM = "first-frame" Fast-WAM; human clips train the video loss only (has_action=False)
#   batch    : PER_DEV x NUM_GPUS x GA = 64 always (= 32 robot + 32 human, exact per GPU); plate by OOM level
#              (state file $PREP_DIR/oom_level): L0 pd16/GA1  L1 pd16/GA1+grad-ckpt  L2 pd8/GA2+grad-ckpt
#              L3 pd4/GA4+grad-ckpt. A non-zero exit whose log carries a memory-failure signature bumps the
#              level for the next pod attempt (backoffLimit). $OOM_FALLBACK (yaml env) = floor, $OOM_LEVEL_OVERRIDE forces.
#   ckpt     : $MODEL_OUTPUT_DIR/<run>/checkpoints/{weights/step_NNNNNN.pt, state/step_NNNNNN/} every 1000, keep 5
#   resume   : automatic — newest state/step_* with trainer_state.json -> resume=<dir> (sample-based position)
#   one-time : (a) ActionDiT backbone .pt — SHARED with the DROID run: the payload holds only the action_dim-
#              independent backbone (blocks/norms/text+time embedders; action_encoder/head are skipped), so the
#              8-dim DROID file is valid for 28 dims (b) dataset_stats.true_global.json — TRUE global q01/q99 over
#              robot frames only (scripts/compute_openarm_true_global_stats.py; marker huiwon_true_global_quantiles;
#              the ONLY stats file the dataset class accepts) (c) umT5 text cache for every tasks.jsonl row of the
#              11 subsets (torchrun, all GPUs). Each runs only when its output is missing.
#   logs     : /data/huiwon/logs/gr00t-vdm/<TAG>-<ts>.{out,err} (or the caller's $FASTWAM_LOG_PREFIX)
# =====================================================================================================
set -uo pipefail

TAG="${TAG:-fastwam-openarm-wam-v1human-rh11-b64-50k-4gpu}"
RUN_NAME="${RUN_NAME:-fastwam_openarm_wam_v1human_rh11_b64_50k_4gpu}"
TASK="${TASK:-openarm_fastwam_5b_b64_50k}"
BASE_DIR="${BASE_DIR:-/data/huiwon/fastwam-openarm}"
LOG_DIR="${LOG_DIR:-/data/huiwon/logs/gr00t-vdm}"
PREP_DIR="${PREP_DIR:-/data/huiwon/data/openarm_fastwam}"
CKPT_BASE="${CKPT_BASE:-/data/huiwon/checkpoints/fastwam_base}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/data/rlwrld-unified-checkpoints/huiwon/fastwam}"
OA="${OPENARM_DATA_ROOT:-/data/huiwon/data/openarm_wam_v1}"
ROBOT_SETS="$OA/robot/openarm_ego_jungwook $OA/robot/openarm_teleop_v3/bottle $OA/robot/openarm_teleop_v3/cup $OA/robot/openarm_teleop_v3/doll $OA/robot/openarm_teleop_v3/snack $OA/robot/banana_v21_openarm28"
HUMAN_SETS="$OA/human_as_openarm28/rlwrld_human_lerobot $OA/human_as_openarm28/openarm_validation_v2_junhyeong/close_air_fryer $OA/human_as_openarm28/openarm_validation_v2_junhyeong/left_hand_box_white_container $OA/human_as_openarm28/openarm_validation_v2_junhyeong/open_air_fryer $OA/human_as_openarm28/anyh2r"

mkdir -p "$LOG_DIR"
if [ -z "${FASTWAM_LOG_PREFIX:-}" ]; then
  L="$LOG_DIR/$TAG-$(date +%Y%m%d_%H%M%S)"
  exec > >(tee -a "$L.out") 2> >(tee -a "$L.err" >&2)
else
  L="$FASTWAM_LOG_PREFIX"
fi
echo "[$TAG] launcher START $(date -u +%FT%TZ) host=$(hostname) log=$L.{out,err}"
fatal() { echo "[$TAG] FATAL: $*"; echo "[$TAG] rc=1 $(date -u +%FT%TZ)"; exit 1; }
OOM_RE='CUDA out of memory|OutOfMemoryError|std::bad_alloc|Killed'
OOM_LEVEL_FILE="$PREP_DIR/oom_level"
OOM_LEVEL_MAX=3
export WANDB_ENTITY="${WANDB_ENTITY:-huiwoen0516}" WANDB_PROJECT=fastwam WANDB_RUN_ID="$RUN_NAME" WANDB_RESUME=allow
WANDB_MODE_OVERRIDE=()
if [ -z "${WANDB_API_KEY:-}" ] || [[ "${WANDB_API_KEY:-}" == REPLACE_WITH* ]]; then
  echo "[$TAG] WARN: no usable WANDB_API_KEY -> wandb.mode=offline"
  export WANDB_MODE=offline; WANDB_MODE_OVERRIDE=("wandb.mode=offline")
fi
set -x

# ── batch plate, chosen by the OOM level ────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
mkdir -p "$PREP_DIR"
LEVEL_FROM_FILE=0
if [ -s "$OOM_LEVEL_FILE" ]; then
  v="$(tr -dc '0-9' < "$OOM_LEVEL_FILE")"; [ -n "$v" ] && LEVEL_FROM_FILE=$((10#$v))
fi
OOM_LEVEL=$LEVEL_FROM_FILE
LEVEL_FLOOR="${OOM_FALLBACK:-0}"; [[ "$LEVEL_FLOOR" =~ ^[0-9]+$ ]] || fatal "OOM_FALLBACK must be an integer"
[ "$LEVEL_FLOOR" -gt "$OOM_LEVEL" ] && OOM_LEVEL=$LEVEL_FLOOR
if [ -n "${OOM_LEVEL_OVERRIDE:-}" ]; then OOM_LEVEL=$((10#$OOM_LEVEL_OVERRIDE)); fi
[ "$OOM_LEVEL" -le "$OOM_LEVEL_MAX" ] || fatal "OOM level $OOM_LEVEL > max $OOM_LEVEL_MAX (state file $OOM_LEVEL_FILE)"
case "$OOM_LEVEL" in
  0) PER_DEV=16; GA=1; GC=false ;;   # default plate: 8 robot + 8 human per GPU
  1) PER_DEV=16; GA=1; GC=true ;;    # + activation checkpointing in every MoT block
  2) PER_DEV=8;  GA=2; GC=true ;;    # half micro-batch (4+4), 2x accumulation
  3) PER_DEV=4;  GA=4; GC=true ;;    # quarter micro-batch (2+2), 4x accumulation
esac
if [ "$GA" -gt 1 ]; then
  export ALLOW_GA_GT1=1
  echo "[$TAG] ############ OOM LEVEL $OOM_LEVEL: GA=$GA > 1 on deepspeed 0.18.7 ZeRO-1 — UNVERIFIED (see DROID launcher note) ############"
fi
echo "[$TAG] OOM level: file=$LEVEL_FROM_FILE floor(OOM_FALLBACK)=$LEVEL_FLOOR override=${OOM_LEVEL_OVERRIDE:-none} -> resolved L$OOM_LEVEL: pd$PER_DEV x GA$GA grad_ckpt=$GC (state file $OOM_LEVEL_FILE, max L$OOM_LEVEL_MAX)"
MAX_STEPS="${MAX_STEPS:-50000}"; SAVE_EVERY="${SAVE_EVERY:-1000}"; SAVE_LIMIT="${SAVE_LIMIT:-5}"; NW="${NW:-12}"
EXPECT_EFF="${EXPECT_EFF:-64}"
EFF=$(( PER_DEV * NUM_GPUS * GA ))
[ "$EFF" -eq "$EXPECT_EFF" ] || fatal "effective batch $EFF != $EXPECT_EFF (pd$PER_DEV x $NUM_GPUS gpu x GA$GA)"
[ $(( PER_DEV % 2 )) -eq 0 ] || fatal "per-device batch $PER_DEV must be even (exact robot/human halves)"
if [ "$GA" -gt 1 ] && [ "${ALLOW_GA_GT1:-0}" != 1 ]; then
  fatal "GA=$GA > 1 refused with the pinned deepspeed (set ALLOW_GA_GT1=1 after verifying / upgrading deepspeed >= 0.19.6)"
fi

# ── environment ─────────────────────────────────────────────────────────────────────────────────────
export DIFFSYNTH_MODEL_BASE_PATH="$CKPT_BASE" DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export FASTWAM_ACTION_DIT_PT="$CKPT_BASE/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
export OPENARM_DATA_ROOT="$OA"
export FASTWAM_OPENARM_TEXT_CACHE="$PREP_DIR/text_embeds_cache"
export FASTWAM_VIDEO_BACKEND="${FASTWAM_VIDEO_BACKEND:-torchcodec}"
STATS_JSON="$PREP_DIR/dataset_stats.true_global.json"         # <- what training reads (marker-gated)
export HF_HOME=/data/huiwon/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/data/huiwon/.cache/hf_datasets_fastwam_openarm
export TMPDIR=/data/huiwon/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=/data/huiwon/.cache/torchinductor_fastwam
mkdir -p "$PREP_DIR" "$HF_DATASETS_CACHE" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

# ── preflight gates ─────────────────────────────────────────────────────────────────────────────────
cd "$BASE_DIR" || fatal "no clone at $BASE_DIR"
echo "[$TAG] repo branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit) HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
PY="$BASE_DIR/.venv/bin/python"
test -x "$PY" || fatal "no venv at $BASE_DIR/.venv (BASE_DIR=$BASE_DIR bash run_scripts/install_venv.sh on the debug pod)"
# shellcheck disable=SC1091
source "$BASE_DIR/.venv/bin/activate"
case "$(python -V 2>&1)" in *" 3.10."*) ;; *) fatal "venv is not python 3.10: $(python -V 2>&1)";; esac
python -c "import fastwam, torch, accelerate, deepspeed, transformers, hydra, os, sys; f=os.path.realpath(fastwam.__file__); assert f.startswith(os.path.realpath('$BASE_DIR')), f; print('imports OK', f, 'torch', torch.__version__, 'deepspeed', deepspeed.__version__, 'accelerate', accelerate.__version__)" || fatal "fastwam env incomplete or imported from the wrong tree"
head -1 "$BASE_DIR/.venv/bin/torchrun" | grep -q "^#!$BASE_DIR/.venv/bin/python" || fatal "torchrun shebang does not point into $BASE_DIR/.venv (hardlinked venv?)"
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')" || fatal "torchcodec unusable (ffmpeg shared libs missing?)"
python -m py_compile scripts/train.py scripts/precompute_text_embeds.py scripts/preprocess_action_dit_backbone.py \
  scripts/compute_openarm_true_global_stats.py scripts/compose_check.py scripts/check_wan22_weights.py \
  src/fastwam/trainer.py src/fastwam/runtime.py src/fastwam/utils/samplers.py src/fastwam/models/wan22/fastwam.py \
  src/fastwam/datasets/lerobot/robot_video_dataset.py src/fastwam/datasets/lerobot/openarm_dataset.py \
  src/fastwam/datasets/lerobot/transforms/openarm.py src/fastwam/datasets/lerobot/processors/openarm_processor.py || fatal "python syntax"
for f in configs/task/$TASK.yaml configs/data/openarm_wan22_5b.yaml configs/model/fastwam_droid.yaml \
         scripts/accelerate_configs/accelerate_zero1_ds.yaml scripts/ds_configs/ds_zero1_config.json; do
  test -s "$f" || fatal "missing config $f"
done
grep -q '"stage": 1' scripts/ds_configs/ds_zero1_config.json || fatal "deepspeed config is not ZeRO-1"
grep -q '"device": "none"' scripts/ds_configs/ds_zero1_config.json || fatal "deepspeed config enables offload"
grep -q "BalancedGroupEpochSampler" src/fastwam/trainer.py || fatal "trainer lacks the balanced robot/human sampler"
grep -q "has_action" src/fastwam/models/wan22/fastwam.py || fatal "model lacks the has_action loss mask"
grep -q "min_lr_ratio" src/fastwam/trainer.py || fatal "trainer lacks the min_lr_ratio / warmup_steps knobs"
for d in $ROBOT_SETS $HUMAN_SETS; do
  test -s "$d/meta/info.json" && test -s "$d/meta/tasks.jsonl" && test -d "$d/videos/chunk-000" && test -d "$d/data/chunk-000" || fatal "dataset root unreadable: $d"
done
for d in $HUMAN_SETS; do grep -q '"human": true' "$d/meta/wam_human.json" || fatal "$d is not marked human"; done
for d in $ROBOT_SETS; do test -f "$d/meta/wam_human.json" && fatal "$d is marked human but listed as robot"; done
test -f "$OA/robot/banana_v21_openarm28/meta/wam_action_groups.json" || fatal "banana lacks meta/wam_action_groups.json"
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
  echo "[$TAG] prep(a): ActionDiT backbone present ($(du -h "$FASTWAM_ACTION_DIT_PT" | cut -f1); action_dim-independent payload shared with DROID)"
fi
python -c "
import torch,sys; p=torch.load(sys.argv[1], map_location='cpu', weights_only=False); ks=list(p['backbone_state_dict'])
bad=[k for k in ks if k.startswith('action_encoder') or k.startswith('head')]; assert not bad, bad; print('ActionDiT payload OK:', len(ks), 'backbone tensors, no action_encoder/head')" "$FASTWAM_ACTION_DIT_PT" || fatal "ActionDiT payload carries action_dim-dependent tensors"
if [ ! -s "$STATS_JSON" ]; then
  echo "[$TAG] prep(b): TRUE global q01/q99 over robot frames (human excluded, banana absent groups excluded) -> $STATS_JSON"
  ROOT_ARGS=(); for d in $ROBOT_SETS $HUMAN_SETS; do ROOT_ARGS+=(--root "$d"); done
  python scripts/compute_openarm_true_global_stats.py "${ROOT_ARGS[@]}" --out-dir "$PREP_DIR" \
    --staging-dir "$TMPDIR/fastwam_openarm_trueq_$$" --horizon 24 --anchor-stride 1 --workers 8 --collapse-range-below 0.02 || fatal "openarm true global stats"
else
  echo "[$TAG] prep(b): stats present: $STATS_JSON"
fi
python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('huiwon_true_global_quantiles') is True and len(d['action']['default']['global_q01'])==28 and d['huiwon_true_global_quantiles_meta'].get('skipped_human_roots') else 1)" "$STATS_JSON" \
  || fatal "$STATS_JSON lacks the marker huiwon_true_global_quantiles=true / 28 dims / human-exclusion meta; rebuild with scripts/compute_openarm_true_global_stats.py"
echo "[$TAG] stats: TRUE global q01/q99 (robot-only, huiwon override) from $STATS_JSON"
export FASTWAM_OPENARM_STATS="$STATS_JSON"
TXT_DONE="$FASTWAM_OPENARM_TEXT_CACHE/.precompute_done"
if [ ! -f "$TXT_DONE" ]; then
  echo "[$TAG] prep(c): umT5-XXL text-embedding cache for every meta/tasks.jsonl prompt of the 11 subsets -> $FASTWAM_OPENARM_TEXT_CACHE"
  mkdir -p "$FASTWAM_OPENARM_TEXT_CACHE"
  torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/precompute_text_embeds.py task="$TASK" +overwrite=false \
    && touch "$TXT_DONE" || fatal "text embedding precompute"
else
  echo "[$TAG] prep(c): text cache present ($(ls "$FASTWAM_OPENARM_TEXT_CACHE" | wc -l) files)"
fi
NPROMPT=$(for d in $ROBOT_SETS $HUMAN_SETS; do cat "$d/meta/tasks.jsonl"; done | python -c "import sys,json; print(len({json.loads(l)['task'] for l in sys.stdin if l.strip()}))")
NCACHE=$(ls "$FASTWAM_OPENARM_TEXT_CACHE" | grep -c '\.t5_len128\.wan22ti2v5b\.pt$')
[ "$NCACHE" -ge "$NPROMPT" ] || fatal "text cache has $NCACHE files < $NPROMPT distinct prompts"

# ── compose gate (hydra only; proves the resolved config carries the plate above) ───────────────────
python scripts/compose_check.py --task "$TASK" --world "$NUM_GPUS" --expect-effective "$EXPECT_EFF" --check-paths --check-meta \
  --expect-fps 20,30 --expect-raw-dims 28,28 --expect-proc-dims 28,28 --expect-layout vertical --expect-video-size 384x256 \
  --expect-lr 1e-4 --expect-wd 0.01 --expect-warmup 2500 --expect-min-lr-ratio 0.1 --expect-max-steps "$MAX_STEPS" \
  --expect-frames 25/3 --expect-group-fractions 0.5,0.5 --expect-relative-dims 2,16 --expect-aug moderate -- \
  batch_size="$PER_DEV" gradient_accumulation_steps="$GA" model.mot_checkpoint_mixed_attn="$GC" \
  max_steps="$MAX_STEPS" save_every="$SAVE_EVERY" save_total_limit="$SAVE_LIMIT" num_workers="$NW" \
  learning_rate=1e-4 warmup_steps=2500 min_lr_ratio=0.1 weight_decay=1e-2 || fatal "compose gate (unification self-check)"

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

echo "[$TAG] BANNER oom_level=L$OOM_LEVEL eff_batch=$EFF = ${NUM_GPUS}gpu x pd${PER_DEV} x GA${GA} (robot $((PER_DEV/2)) + human $((PER_DEV/2)) per GPU) | grad_ckpt=$GC | lr=1e-4 warmup=2500 min_lr=0.1xLR wd=0.01 aug=moderate(0.9/5/0.2) | max_steps=$MAX_STEPS save_every=$SAVE_EVERY keep=$SAVE_LIMIT nw=$NW | stats=true-global-q01q99(robot-only, huiwon override) | resume=${RESUME_DIR:-fresh(step 0)} | out=$OUTPUT_DIR"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "$NUM_GPUS" --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${MASTER_PORT:-29500}" \
  scripts/train.py \
  task="$TASK" \
  output_dir="$OUTPUT_DIR" \
  batch_size="$PER_DEV" gradient_accumulation_steps="$GA" model.mot_checkpoint_mixed_attn="$GC" \
  max_steps="$MAX_STEPS" save_every="$SAVE_EVERY" save_total_limit="$SAVE_LIMIT" num_workers="$NW" \
  learning_rate=1e-4 warmup_steps=2500 min_lr_ratio=0.1 weight_decay=1e-2 \
  wandb.name="$RUN_NAME" \
  ${WANDB_MODE_OVERRIDE[@]+"${WANDB_MODE_OVERRIDE[@]}"} \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  "$@"
rc=$?
set +x
if [ "$rc" -ne 0 ]; then
  sync; sleep 3
  HIT="$(grep -a -m1 -E "$OOM_RE" "$L.err" "$L.out" 2>/dev/null | head -n1 | cut -c1-300)"
  if [ -n "$HIT" ]; then
    if [ "$OOM_LEVEL" -lt "$OOM_LEVEL_MAX" ]; then
      NEXT=$((OOM_LEVEL + 1))
      printf '%s\n' "$NEXT" > "$OOM_LEVEL_FILE.tmp" && mv -f "$OOM_LEVEL_FILE.tmp" "$OOM_LEVEL_FILE"
      echo "$(date -u +%FT%TZ) rc=$rc L$OOM_LEVEL->L$NEXT log=$L signature=${HIT}" >> "$OOM_LEVEL_FILE.log"
      echo "[$TAG] OOM-ESCALATE: memory-failure signature found (rc=$rc) at L$OOM_LEVEL -> wrote L$NEXT to $OOM_LEVEL_FILE; the next pod attempt resumes from the newest checkpoint with the L$NEXT plate"
    else
      echo "$(date -u +%FT%TZ) rc=$rc L$OOM_LEVEL (max, no escalation) log=$L signature=${HIT}" >> "$OOM_LEVEL_FILE.log"
      echo "[$TAG] OOM-ESCALATE: memory-failure signature found (rc=$rc) but already at max level L$OOM_LEVEL_MAX -> no escalation; intervene by hand"
    fi
  else
    echo "[$TAG] rc=$rc without a memory-failure signature in $L.{err,out} -> OOM level stays L$OOM_LEVEL"
  fi
fi
echo "[$TAG] rc=$rc $(date -u +%FT%TZ)"
exit $rc
