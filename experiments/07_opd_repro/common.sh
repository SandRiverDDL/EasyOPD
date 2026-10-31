#!/bin/bash
# ============================================================
# 07_opd_repro: shared single-node pipeline (train -> merge -> eval); N_GPUS=8 or 4 (SLOT picks the GPU half).
#
# Based on experiments/06_general_opd/methods/g_opd/launch.sh (same Ray restart,
# resume-by-largest-step, merge-every-ckpt, eval-every-ckpt flow). Changes vs that
# script, one by one:
#   1. paths: real NLU17 paths + /root/envs/easyopd instead of /path/to placeholders
#   2. data: DAPO-Math-17k deduped (scripts/prepare_data.py), not mixed_math_code_10k
#   3. student: Qwen3-1.7B-Base with <|im_end|> as EOS (scripts/make_chat_student.py)
#   4. method switch: +actor_rollout_ref.actor.opd.method=<m> (unified [EasyOPD:OPD]
#      path, which also creates the teacher ref worker) instead of
#      policy_loss.only_reverse_kl_advantages; method knobs come in as OPD_ARGS
#   5. chat template: enable_thinking=False for student rollout and teacher scoring
#   6. merge: verl.model_merger (official) instead of _shared/scripts/merge_fsdp.py
#   7. eval: scripts/evaluate.py (AIME24/25, AMC23, MATH-500, avg@k/pass@k)
#   8. artifacts: runs/train/07_opd_repro/<batch>/<tag>/ on NLU17 (<method>/<tag>/ before 10-31)
#
# A method launch.sh sets METHOD and OPD_ARGS (array) plus any overrides, then
# sources this file. Every variable below can be overridden from the environment.
# Unverified on GPU as of 2026-10-29 (the dev node's GPUs were busy).
# ============================================================
set -euo pipefail
trap 'rc=$?; echo "[FATAL] exited with code $rc at line $LINENO (last cmd: $BASH_COMMAND)" >&2' ERR

: "${METHOD:?METHOD must be set by the method launch.sh}"
declare -p OPD_ARGS >/dev/null 2>&1 || OPD_ARGS=()
declare -p EXTRA_ARGS >/dev/null 2>&1 || EXTRA_ARGS=()

NLU=/mnt/NLU_disk17/chenhaoyu02
ENV_DIR=${ENV_DIR:-/root/envs/easyopd}
PYTHON="${ENV_DIR}/bin/python"
RAY="${ENV_DIR}/bin/ray"
export PATH="${ENV_DIR}/bin:${PATH}"

# Code must be on the local disk (CLAUDE.md: no PYTHONPATH on NFS).
EASYOPD_ROOT=${EASYOPD_ROOT:-/root/EasyOPD}
[ -d "${EASYOPD_ROOT}/easyopd" ] || { echo "EASYOPD_ROOT=${EASYOPD_ROOT} is not an EasyOPD checkout (copy the repo to local disk first)"; exit 1; }
export PYTHONPATH="${EASYOPD_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true NCCL_DEBUG=WARN HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0
unset RAY_ADDRESS

EXP_SCRIPTS="${EASYOPD_ROOT}/experiments/07_opd_repro/scripts"

# ----------------- Models / data -----------------
STUDENT_MODEL=${STUDENT_MODEL:-${NLU}/models/Qwen3-1.7B-Base-imend}
TEACHER_MODEL=${TEACHER_MODEL:-${NLU}/models/Qwen3-4B}
DATA_DIR=${DATA_DIR:-${NLU}/data/opd_repro}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_DIR}/train_en.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_DIR}/math500_dev200.parquet}   # stratified-by-level 200 of math500 (seed 0); rest = math500_test300
REWARD_FN="${EXP_SCRIPTS}/reward_fn.py"

# ----------------- Shared hyperparameters (identical across methods unless a launch.sh overrides) ---
N_GPUS=${N_GPUS:-8}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-4096}
MAX_MODEL_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}          # prompts per step
ROLLOUT_N=${ROLLOUT_N:-4}                         # samples per prompt
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}  # prompts; == batch -> 1 on-policy update / step
PPO_EPOCHS=${PPO_EPOCHS:-1}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
CLIP_LOW=${CLIP_LOW:-0.2}
CLIP_HIGH=${CLIP_HIGH:-0.2}
ACTOR_LR=${ACTOR_LR:-1e-6}  # 10-30 changed from 2e-6 to match common OPD/RL recipes (e.g. Semi-OPD Table 2); runs before 10-30 used 2e-6
LR_WARMUP_RATIO=${LR_WARMUP_RATIO:-0.0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-200}
# Two ckpt kinds, each with its own frequency and retention ([EasyOPD:CKPT] in ray_trainer.py):
#   full    = model + optimizer + extra, resumable; ${FSDP_CKPT_DIR}/global_step_N
#   weights = model shards only, for merge/eval;    ${FSDP_CKPT_DIR}/weights/global_step_N (never resumed from)
# SAVE_WEIGHTS_FREQ <= 0 keeps the original behaviour (full ckpt every SAVE_FREQ steps + the last step).
# With weights on, the last step always gets a weights ckpt, and a full one only if SAVE_FREQ divides it.
SAVE_FREQ=${SAVE_FREQ:-50}                         # full ckpt every N steps; -1 = off
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-null}               # newest full ckpts kept on local disk; null = all
SAVE_WEIGHTS_FREQ=${SAVE_WEIGHTS_FREQ:--1}         # weights-only ckpt every N steps; -1 = off
MAX_WEIGHTS_KEEP=${MAX_WEIGHTS_KEEP:-null}         # newest weights ckpts kept on local disk; null = all (merged hf/ copies are never deleted)
TEST_FREQ=${TEST_FREQ:-10}
# In-training validation (decoupled from rollout length via [EasyOPD:VAL_LEN]); matches the offline eval sampling.
VAL_MAX_TOKENS=${VAL_MAX_TOKENS:-8192}
VAL_N=${VAL_N:-1}                                 # dev200 x 1: trend only (~2.4 pt std); final numbers come from the offline eval
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}  # 0.8 peaked at 76.1/80 GB reserved (1029_2240_smoke_mem08); 0.75 keeps ~7 GB headroom
ENABLE_GRAD_CKPT=${ENABLE_GRAD_CKPT:-True}
FSDP_PARAM_OFFLOAD=${FSDP_PARAM_OFFLOAD:-False}
FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
# Actor master-weight dtype. AdamW m/v are zeros_like(param), so they follow it.
# fp32 = verl's actor default (mixed precision: bf16 compute, fp32 master/m/v).
# bf16 was used by all runs before 10-30; lr~2e-6 updates get rounded away in bf16.
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-fp32}
SEED=${SEED:-1}
DUMP_FREQ=${DUMP_FREQ:-10}                         # token-level parquet every N steps (easyopd/opd/analysis.py); 0 = off

# ----------------- Eval protocol (scripts/evaluate.py) -----------------
EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-aime24:32,aime25:32,amc23:32,math500:8}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.6}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-8192}
EVAL_STEPS=${EVAL_STEPS:-all}                     # "all", "last", or space-separated step numbers
EVAL_BASELINES=${EVAL_BASELINES:-0}               # 1: also eval the student init and the teacher

# ----------------- Run dir layout (runs/INDEX.md: train/<name>/) -----------------
RUN_TAG=${RUN_TAG:-$(date +%m%d_%H%M)_${METHOD}}
# train_scheduler.py always passes RUN_DIR (= <batch>/<tag>); this default is for hand launches (BATCH falls back to METHOD).
RUN_DIR=${RUN_DIR:-${NLU}/runs/train/07_opd_repro/${BATCH:-${METHOD}}/${RUN_TAG}}
# Training ckpts go to the node's local disk; merged HF copies, logs and eval stay on NLU17.
# Local ckpts die with the instance, so Step 4 (merge) must run before the node is released.
FSDP_CKPT_DIR=${FSDP_CKPT_DIR:-/root/easyopd_ckpt/07_opd_repro/${METHOD}/${RUN_TAG}/fsdp}
HF_CKPT_DIR="${RUN_DIR}/hf"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/eval"
mkdir -p "${FSDP_CKPT_DIR}" "${HF_CKPT_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"
# verl's tensorboard logger reads this (verl/utils/tracking.py); per-step scalars land next to the run.
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}
EXPECTED_FINAL_STEP=${TOTAL_STEPS}

RAY_NODE_IP="${RAY_NODE_IP:-$(hostname -I | awk '{print $1}')}"
# Several experiments can share one node (e.g. N_GPUS=4 -> SLOT=0 uses GPUs 0-3, SLOT=1 uses GPUs 4-7).
# Each SLOT gets its own GPU ids, Ray ports and Ray temp dir, and only ever stops its own Ray.
SLOT="${SLOT:-0}"
if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS=$(seq -s, $(( SLOT * N_GPUS )) $(( SLOT * N_GPUS + N_GPUS - 1 )))
fi
[ "$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)" -eq "${N_GPUS}" ] || { echo "GPU_IDS=${GPU_IDS} must list exactly N_GPUS=${N_GPUS} ids" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"        # set before `ray start` so Ray only sees these GPUs
# Base port 7379 (not 6379): some nodes still have a leftover Ray head on 6379 from an
# earlier job, and `ray start` on that port aborts with a session-name assertion.
RAY_PORT="${RAY_PORT:-$(( 7379 + SLOT * 100 ))}"
RAY_TMP="${RAY_TMP:-/tmp/ray_07_slot${SLOT}}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$(( N_GPUS * 8 ))}"

# verl splits (batch * n) and (mini_batch * n) evenly over the GPUs.
(( (TRAIN_BATCH_SIZE * ROLLOUT_N) % N_GPUS == 0 )) || { echo "TRAIN_BATCH_SIZE*ROLLOUT_N=$((TRAIN_BATCH_SIZE*ROLLOUT_N)) not divisible by N_GPUS=${N_GPUS}" >&2; exit 1; }
(( (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % N_GPUS == 0 )) || { echo "PPO_MINI_BATCH_SIZE*ROLLOUT_N=$((PPO_MINI_BATCH_SIZE*ROLLOUT_N)) not divisible by N_GPUS=${N_GPUS}" >&2; exit 1; }

echo "[$(date)] 07_opd_repro method=${METHOD} run_dir=${RUN_DIR}"
echo "  gpus=${GPU_IDS} (N_GPUS=${N_GPUS}, SLOT=${SLOT}) ray=${RAY_NODE_IP}:${RAY_PORT} ray_tmp=${RAY_TMP}"
echo "  student=${STUDENT_MODEL}"
echo "  teacher=${TEACHER_MODEL}"
echo "  train=${TRAIN_PARQUET} batch=${TRAIN_BATCH_SIZE}x${ROLLOUT_N} mini=${PPO_MINI_BATCH_SIZE} epochs=${PPO_EPOCHS} lr=${ACTOR_LR} steps=${TOTAL_STEPS}"
echo "  OPD_ARGS: ${OPD_ARGS[*]:-}"
echo "  EXTRA_ARGS: ${EXTRA_ARGS[*]:-}"

# ============================================================
# Step 1: data parquets + chat-ready student view (idempotent)
# ============================================================
if [ ! -f "${TRAIN_PARQUET}" ] || [ ! -f "${DATA_DIR}/aime24.parquet" ]; then
    echo "[$(date)] ===== Step 1: building parquets in ${DATA_DIR} ====="
    _en_flag=""
    [[ "${TRAIN_PARQUET}" == *_en.parquet ]] && _en_flag="--english_only"
    ${PYTHON} "${EXP_SCRIPTS}/prepare_data.py" --out_dir "${DATA_DIR}" ${_en_flag}
fi
if [[ "${STUDENT_MODEL}" == *-imend ]] && [ ! -f "${STUDENT_MODEL}/tokenizer_config.json" ]; then
    echo "[$(date)] ===== Step 1: building chat-ready student view ${STUDENT_MODEL} ====="
    ${PYTHON} "${EXP_SCRIPTS}/make_chat_student.py" "${STUDENT_MODEL%-imend}" "${STUDENT_MODEL}"
fi

# ============================================================
# Step 2: (re)start Ray on this node (same as 06 g_opd launch.sh)
# ============================================================
RAY_HEAD_ADDRESS="${RAY_NODE_IP}:${RAY_PORT}"
RAY_READY_RETRIES="${RAY_READY_RETRIES:-12}"

# Stop only this slot's Ray (matched by its private temp dir); never `ray stop` / global pkill,
# which would also kill the other experiment on the same node.
stop_own_ray() {
    pkill -9 -f -- "${RAY_TMP}/" 2>/dev/null || true
    sleep 3
}

start_ray() {
    if [ "${SKIP_RAY_RESTART:-0}" != "1" ]; then
        echo "[$(date)] ===== Step 2: restarting Ray (slot ${SLOT}) ====="
        unset RAY_ADDRESS
        stop_own_ray
        command rm -rf "${RAY_TMP}" 2>/dev/null || true
        local wp=$(( 20000 + SLOT * 2000 ))
        ${RAY} start --head --disable-usage-stats --node-ip-address="${RAY_NODE_IP}" --port="${RAY_PORT}" \
            --temp-dir="${RAY_TMP}" --num-cpus="${RAY_NUM_CPUS}" --num-gpus="${N_GPUS}" --include-dashboard=false \
            --node-manager-port=$(( RAY_PORT + 1 )) --object-manager-port=$(( RAY_PORT + 2 )) \
            --ray-client-server-port=$(( RAY_PORT + 3 )) --runtime-env-agent-port=$(( RAY_PORT + 4 )) \
            --metrics-export-port=$(( RAY_PORT + 5 )) --dashboard-agent-listen-port=$(( RAY_PORT + 6 )) \
            --dashboard-agent-grpc-port=$(( RAY_PORT + 7 )) --min-worker-port="${wp}" --max-worker-port=$(( wp + 1999 ))
    fi
    export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
    local attempt
    for attempt in $(seq 1 "${RAY_READY_RETRIES}"); do
        if timeout 30 ${PYTHON} - <<'PYEOF'
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, log_to_driver=False)
@ray.remote
def ping():
    return "ok"
assert ray.get(ping.remote(), timeout=10) == "ok"
print(f"[RayCheck] {os.environ['RAY_ADDRESS']} ping ok")
os._exit(0)
PYEOF
        then
            return 0
        fi
        echo "[$(date)] [RayCheck] attempt ${attempt}/${RAY_READY_RETRIES} failed; retrying..."
        sleep 5
    done
    echo "[$(date)] ERROR: Ray did not pass the remote ping check." >&2
    exit 1
}

# ============================================================
# Step 3: train (resume from the largest global_step_* via trainer.resume_mode=auto)
# ============================================================
largest_step() {
    local best=0 c s
    shopt -s nullglob
    for c in "${FSDP_CKPT_DIR}"/global_step_* "${FSDP_CKPT_DIR}"/weights/global_step_*; do
        s=${c##*/global_step_}
        [[ "${s}" =~ ^[0-9]+$ ]] && (( s > best )) && best=${s}
    done
    shopt -u nullglob
    echo "${best}"
}

LARGEST_EXISTING_STEP=$(largest_step)
if [ "${FORCE_RETRAIN:-0}" = "1" ]; then
    SKIP_TRAIN=0
elif [ "${SKIP_TRAIN:-0}" = "1" ] || [ "${LARGEST_EXISTING_STEP}" -ge "${EXPECTED_FINAL_STEP}" ]; then
    echo "[$(date)] global_step_${LARGEST_EXISTING_STEP} >= ${EXPECTED_FINAL_STEP} (or SKIP_TRAIN=1); skipping Step 3."
    SKIP_TRAIN=1
else
    SKIP_TRAIN=0
fi

ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$(( MAX_MODEL_LEN > 16384 ? MAX_MODEL_LEN : 16384 ))}
# OPD_METHOD defaults to METHOD. Set it to the empty string for plain RL baselines (grpo / dapo):
# no actor.opd.method -> no ref worker (teacher never loaded), no OPD hooks; the verifier reward drives GRPO.
OPD_METHOD=${OPD_METHOD-${METHOD}}
_OPD_METHOD_ARG=()
if [ -n "${OPD_METHOD}" ]; then _OPD_METHOD_ARG=( +actor_rollout_ref.actor.opd.method="${OPD_METHOD}" ); fi
RESUME_MODE=auto
[ "${FORCE_RETRAIN:-0}" = "1" ] && RESUME_MODE=disable

if [ "${SKIP_TRAIN}" = "0" ]; then
    start_ray
    echo "[$(date)] ===== Step 3: train ${METHOD} (largest existing step ${LARGEST_EXISTING_STEP}, resume_mode=${RESUME_MODE}) ====="
    TRAIN_LAUNCH_CWD="/tmp/easyopd_07_${METHOD}_${RUN_TAG}"
    mkdir -p "${TRAIN_LAUNCH_CWD}"
    (
    cd "${TRAIN_LAUNCH_CWD}"
    ${PYTHON} -m verl.trainer.main_ppo \
        +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        "data.train_files=['${TRAIN_PARQUET}']" \
        "data.val_files=['${VAL_PARQUET}']" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.max_prompt_length="${MAX_PROMPT_LEN}" \
        data.max_response_length="${MAX_RESPONSE_LEN}" \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.shuffle=True \
        +data.seed="${SEED}" \
        data.prompt_key=prompt \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="${STUDENT_MODEL}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRAD_CKPT}" \
        +actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
        actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO}" \
        actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
        actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
        actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
        actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
        actor_rollout_ref.actor.fsdp_config.param_offload="${FSDP_PARAM_OFFLOAD}" \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload="${FSDP_OPTIMIZER_OFFLOAD}" \
        "${_OPD_METHOD_ARG[@]}" \
        "${OPD_ARGS[@]}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
        actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
        actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
        actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
        actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
        actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
        actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
        actor_rollout_ref.rollout.val_kwargs.response_length="${VAL_MAX_TOKENS}" \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
        actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
        +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
        +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
        custom_reward_function.path="${REWARD_FN}" \
        custom_reward_function.name=compute_score \
        trainer.balance_batch=True \
        'trainer.logger=["console","tensorboard"]' \
        trainer.project_name=easyopd-07-opd-repro \
        trainer.experiment_name="${METHOD}_${RUN_TAG}" \
        trainer.n_gpus_per_node="${N_GPUS}" \
        trainer.nnodes=1 \
        trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.total_training_steps="${TOTAL_STEPS}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.max_actor_ckpt_to_keep="${MAX_CKPT_KEEP}" \
        +trainer.save_weights_freq="${SAVE_WEIGHTS_FREQ}" \
        +trainer.max_weights_ckpt_to_keep="${MAX_WEIGHTS_KEEP}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.resume_mode="${RESUME_MODE}" \
        trainer.default_local_dir="${FSDP_CKPT_DIR}" \
        trainer.validation_data_dir="${RUN_DIR}/val_gen" \
        +actor_rollout_ref.actor.opd.dump_freq="${DUMP_FREQ}" \
        +actor_rollout_ref.actor.opd.dump_dir="${RUN_DIR}/token_dump" \
        "${EXTRA_ARGS[@]}" \
        2>&1
    ) | tee -a "${LOG_DIR}/train.log"
    stop_own_ray
    echo "[$(date)] ===== Step 3 done ====="
fi

# ============================================================
# Step 4: merge every global_step_X/actor -> hf/global_step_X (official verl merger)
# Sources: weights/global_step_X (preferred) and full global_step_X; one merge per step. The merger only
# reads model shards + fsdp_config.json + huggingface/, which weights-only ckpts also have.
# ============================================================
shopt -s nullglob
ALL_CKPTS=()
while IFS= read -r _line; do ALL_CKPTS+=( "${_line}" ); done < <(
    { for c in "${FSDP_CKPT_DIR}"/weights/global_step_*; do echo "${c##*/global_step_} 0 ${c}"; done
      for c in "${FSDP_CKPT_DIR}"/global_step_*; do echo "${c##*/global_step_} 1 ${c}"; done
    } | sort -k1,1n -k2,2n | awk '!seen[$1]++ {print $3}')
shopt -u nullglob
[ ${#ALL_CKPTS[@]} -gt 0 ] || { echo "[$(date)] ERROR: no checkpoint in ${FSDP_CKPT_DIR}" >&2; exit 1; }

merged_ok() { compgen -G "$1/*.safetensors" >/dev/null || [ -f "$1/pytorch_model.bin" ]; }

echo "[$(date)] ===== Step 4: merging ${#ALL_CKPTS[@]} checkpoint(s) ====="
for CKPT in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT}")
    TARGET="${HF_CKPT_DIR}/${STEP_NAME}"
    [ -d "${CKPT}/actor" ] || { echo "[${STEP_NAME}] no actor/, skip"; continue; }
    merged_ok "${TARGET}" && { echo "[${STEP_NAME}] already merged"; continue; }
    ${PYTHON} -m verl.model_merger merge --backend fsdp --local_dir "${CKPT}/actor" --target_dir "${TARGET}" \
        2>&1 | tee "${LOG_DIR}/merge_${STEP_NAME}.log"
    # The merger copies tokenizer + generation_config from actor/huggingface (= the student dir),
    # so the <|im_end|> EOS survives; check it so eval cannot silently fall back to <|endoftext|>.
    ${PYTHON} -c "import sys,json; g=json.load(open(sys.argv[1])); e=g['eos_token_id']; e=e if isinstance(e,list) else [e]; assert 151645 in e, e" \
        "${TARGET}/generation_config.json"
done

# ============================================================
# Step 5: eval (scripts/evaluate.py: avg@k / pass@k on AIME24/25, AMC23, MATH-500)
# ============================================================
eval_done() { [ -f "${RESULTS_DIR}/results.jsonl" ] && grep -q "\"tag\": \"$1\"" "${RESULTS_DIR}/results.jsonl"; }

run_eval() {
    local model="$1" tag="$2"
    if [ "${FORCE_REEVAL:-0}" != "1" ] && eval_done "${tag}"; then
        echo "[$(date)] [eval] ${tag} already in results.jsonl, skip"; return 0
    fi
    echo "[$(date)] [eval] ${tag} <- ${model}"
    ${PYTHON} "${EXP_SCRIPTS}/evaluate.py" --model_path "${model}" --data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}" --tag "${tag}" --benchmarks "${EVAL_BENCHMARKS}" \
        --temperature "${EVAL_TEMPERATURE}" --top_p "${EVAL_TOP_P}" --max_tokens "${EVAL_MAX_TOKENS}" \
        --max_prompt_len "${MAX_PROMPT_LEN}" --dp_size "${N_GPUS}" --gpu_ids "${GPU_IDS}" \
        2>&1 | tee "${LOG_DIR}/eval_${tag}.log"
}

echo "[$(date)] ===== Step 5: eval (EVAL_STEPS=${EVAL_STEPS}) ====="
if [ "${EVAL_BASELINES}" = "1" ]; then
    run_eval "${STUDENT_MODEL}" "student_init"
    run_eval "${TEACHER_MODEL}" "teacher"
fi
for CKPT in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT}")
    STEP=${STEP_NAME#global_step_}
    case "${EVAL_STEPS}" in
        all) ;;
        last) [ "${CKPT}" = "${ALL_CKPTS[-1]}" ] || continue ;;
        *) [[ " ${EVAL_STEPS} " == *" ${STEP} "* ]] || continue ;;
    esac
    merged_ok "${HF_CKPT_DIR}/${STEP_NAME}" || { echo "[${STEP_NAME}] not merged, skip eval"; continue; }
    run_eval "${HF_CKPT_DIR}/${STEP_NAME}" "${METHOD}_${STEP_NAME}"
done

echo "[$(date)] ===== ${METHOD} done ====="
echo "run dir: ${RUN_DIR}"
echo "results: ${RESULTS_DIR}/results.jsonl"
