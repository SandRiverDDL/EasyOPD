#!/bin/bash
# Runs fork_probe.py gen -> score -> cont on the GPUs of ONE node; shards are independent, so a
# 2-node run is this script started on each node with NODE_RANK=0/1 (global shard = NODE_RANK*N_GPUS + gpu).
# New script; the per-GPU process layout (CUDA_VISIBLE_DEVICES, VLLM_PORT base, VLLM_USE_V1=0) follows evaluate.py.
#   OUT_DIR=... NODE_RANK=0 NUM_NODES=2 N_GPUS=8 CODE=/root/EasyOPD bash run_fork_probe.sh [fork_probe args]
set -uo pipefail
: "${OUT_DIR:?}"
NODE_RANK=${NODE_RANK:-0}; NUM_NODES=${NUM_NODES:-1}; N_GPUS=${N_GPUS:-8}
CODE=${CODE:-/root/EasyOPD}
PY=/root/envs/easyopd/bin/python
SCRIPT="${CODE}/experiments/07_opd_repro/scripts/probes/fork_probe.py"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}" VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
mkdir -p "${OUT_DIR}/logs"
NS=$(( NUM_NODES * N_GPUS ))
pids=()
for g in $(seq 0 $(( N_GPUS - 1 ))); do
    sid=$(( NODE_RANK * N_GPUS + g ))
    log="${OUT_DIR}/logs/shard_$(printf %02d $sid).log"
    (
        export CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$(( 31000 + 50 * g ))
        for stage in gen score cont; do
            echo "[$(date +%T)] shard ${sid} stage ${stage} start"
            "$PY" "$SCRIPT" "$stage" --out_dir "$OUT_DIR" --shard_id "$sid" --num_shards "$NS" "$@" || { echo "[FATAL] shard ${sid} stage ${stage} failed"; exit 1; }
        done
        echo "[$(date +%T)] shard ${sid} DONE"
    ) > "$log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$(date +%T)] node ${NODE_RANK} finished rc=${rc}"
exit $rc
