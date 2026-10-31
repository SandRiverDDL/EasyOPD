#!/bin/bash
# Superseded (10-30) by configs/exps/*.yaml + scripts/train_scheduler.py (easyopd/docs/reference/train-scheduler.md);
# kept only because the 1029_2323_abl80v10 / 1030_0318 runs were started with it.
# 07_opd_repro data / signal ablations (vanilla_opd, 80 steps, dev200 val every 10 steps).
# New wrapper around methods/vanilla_opd/launch.sh (no changes to it); one ablation per call:
#   bash run_ablation.sh <name> <node_ip> <slot> <run_tag_prefix>
# Each call rsyncs the NFS repo to the node's /root/EasyOPD and starts launch.sh with setsid nohup.
# Shared overrides = the 1029_1648_b32n1r2k_80s20 protocol (batch 32 x n=1, response 2048,
# teacher resident) + SAVE_FREQ=80 (only the final ckpt) + EVAL_STEPS=none (offline eval later).
# 1030: + diagnostics (diag_topk / teacher entropy / token groups) for every run; vanilla runs use
# opd.terminal_stop=cross_eos (Base student ends with <|endoftext|>, see docs/pitfalls.md).
# power / dial: official methods at 200 steps, val + ckpt every 25; they keep their official terminal
# masking (terminal_eos_ids now [151643,151645] in their launch.sh) instead of cross_eos.
set -euo pipefail
NAME=$1; NODE=$2; SLOT=$3; PREFIX=$4
NLU=/mnt/NLU_disk17/chenhaoyu02
D=${NLU}/data/opd_repro
TRAIN=${D}/train_en.parquet; SEED=1; EXTRA=""
METHOD=vanilla_opd; STEPS=80; SAVE=80; TEST=10
STU=${NLU}/models/Qwen3-1.7B-Base-imend
DIAG="+actor_rollout_ref.actor.opd.diag_topk=20 +actor_rollout_ref.actor.opd.diag_teacher_entropy=True +actor_rollout_ref.actor.opd.diag_tokenizer=${STU}"
EOSFIX="+actor_rollout_ref.actor.opd.terminal_stop=cross_eos"
case "${NAME}" in
    full)      ;;
    dapo1)     TRAIN=${D}/train_dapo1.parquet ;;
    dapo4)     TRAIN=${D}/train_dapo4.parquet ;;
    dapo16)    TRAIN=${D}/train_dapo16.parquet ;;
    wildchat)  TRAIN=${D}/train_wildchat.parquet ;;
    template)  TRAIN=${D}/train_template_sys.parquet ;;
    sign)      EXTRA="+actor_rollout_ref.actor.opd.adv_transform=sign" ;;
    fixedneg)  EXTRA="+actor_rollout_ref.actor.opd.adv_transform=fixed_neg_lowp" ;;
    seed2)     SEED=2 ;;
    power)     METHOD=power_opd; STEPS=200; SAVE=25; TEST=25; EOSFIX="" ;;
    dial)      METHOD=dial_opd;  STEPS=200; SAVE=25; TEST=25; EOSFIX="" ;;
    *) echo "unknown ablation ${NAME}" >&2; exit 1 ;;
esac
TAG=${PREFIX}_${NAME}
EXTRA="${EXTRA} ${EOSFIX} ${DIAG}"
RD=${NLU}/runs/train/07_opd_repro/${METHOD}/${TAG}
mkdir -p "${RD}/logs"
# Record which code this run used (experiments/ is not committed yet, so keep the diff too).
{ cd ${NLU}/EasyOPD; echo "commit $(git rev-parse HEAD)"; git status --short; git diff; } > "${RD}/code_version.txt" 2>&1 || true
timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=10 "root@${NODE}" "rsync -a --exclude .git --exclude __pycache__ ${NLU}/EasyOPD/ /root/EasyOPD/ && cd /root/EasyOPD && \
N_GPUS=4 SLOT=${SLOT} RUN_TAG=${TAG} TRAIN_PARQUET=${TRAIN} SEED=${SEED} TRAIN_BATCH_SIZE=32 ROLLOUT_N=1 MAX_RESPONSE_LEN=2048 \
REF_PARAM_OFFLOAD=False TOTAL_STEPS=${STEPS} SAVE_FREQ=${SAVE} TEST_FREQ=${TEST} EVAL_STEPS=none \
setsid nohup bash -c 'EXTRA_ARGS=(${EXTRA}); source experiments/07_opd_repro/methods/${METHOD}/launch.sh' \
> ${RD}/logs/launcher.log 2>&1 < /dev/null & sleep 2; ps -eo pid,sid,args | grep '[R]UN_TAG=${TAG}\|[s]ource experiments' | head -1 | cut -c1-60" || true  # ssh may linger on the detached job; it survives the timeout (setsid)
echo "${NAME} -> ${NODE} slot ${SLOT}: ${RD}/logs/launcher.log"
