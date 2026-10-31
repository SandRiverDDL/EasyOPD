#!/usr/bin/env python3
"""Post-training eval scheduler for the 07_opd_repro 80-step round (new code, no upstream reference).

Runs on the driver machine (NFS visible, ssh to the 3 experiment nodes) and keeps every free
4-GPU slot busy with evaluation jobs, in this order:
  1. math500:8 on global_step 80, 60, 40, 20 of every method; a method becomes eligible once its
     launcher reached Step 5. Then math500:8 for student_init / teacher (reference rows).
  2. per method: pick the best ckpt by math500 avg@8 over steps 20/40/60/80 (ties -> later step),
     then run aime24:16, aime25:16, amc23:1 on it.
  3. lowest priority: the same extra benchmarks for student_init and teacher (reference rows).
The scheduler only ever *launches* evaluate.py on the 3 nodes below. It never kills anything.
State is derived from results.jsonl + `ps` on the nodes, so it is safe to restart.
"""
import argparse
import json
import os
import subprocess
import time

NLU = "/mnt/NLU_disk17/chenhaoyu02"
RUN_TAG = "1029_1648_b32n1r2k_80s20"
RUNS = f"{NLU}/runs/train/07_opd_repro"
DATA = f"{NLU}/data/opd_repro"
METHODS = ["vanilla_opd", "power_opd", "dial_opd", "eopd", "revisit_opd", "topd"]
NODES = ["10.206.120.80", "10.206.114.79", "10.206.120.12"]  # the only machines this script touches
GPUS = {0: "0,1,2,3", 1: "4,5,6,7"}
STEPS_TODO = [80, 60, 40, 20]  # 80 included: the launchers' own step-80 evals hit EADDRINUSE and exited
ALL_STEPS = [20, 40, 60, 80]
MATH = "math500:8"
EXT = "aime24:16,aime25:16,amc23:1"
BASELINE_MATH = {"student_init": f"{NLU}/models/Qwen3-1.7B-Base-imend", "teacher": f"{NLU}/models/Qwen3-4B"}
BASELINES = {"student_init_ext": f"{NLU}/models/Qwen3-1.7B-Base-imend", "teacher_ext": f"{NLU}/models/Qwen3-4B"}
MAX_ATTEMPTS = 2
PY = "/root/envs/easyopd/bin/python"

PROBE = r"""
ps -eo pid,args | awk '$2=="bash" && index($3,"experiments/07_opd_repro/methods/")==1 {print $1}' | while read p; do
  s=$(tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -m1 '^SLOT='); echo "L ${s#SLOT=}"; done
ps -eo args | grep -E 'evaluate.py --model_path' | grep -v -e '--worker' -e grep | sed 's/^/E /'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/^/G /'
"""


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_dir(m):
    return f"{RUNS}/{m}/{RUN_TAG}"


def done_tags(m):
    p = f"{run_dir(m)}/eval/results.jsonl"
    out = {}
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
                out[r["tag"]] = r
            except (ValueError, KeyError):
                pass
    return out


def launcher_reached_eval(m):
    p = f"{run_dir(m)}/logs/launcher.log"
    return os.path.exists(p) and "Step 5: eval" in open(p, errors="ignore").read()


def ckpt_ready(m, step):
    d = f"{run_dir(m)}/hf/global_step_{step}"
    return os.path.isdir(d) and any(f.endswith(".safetensors") for f in os.listdir(d))


def probe(ip):
    """Return {'launcher_slots': set, 'eval_gpu_sets': set, 'eval_tags': set, 'mem': {gpu: MiB}} or None."""
    try:
        out = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"root@{ip}", PROBE],
                             capture_output=True, text=True, timeout=40).stdout
    except subprocess.TimeoutExpired:
        return None
    st = {"launcher_slots": set(), "eval_gpu_sets": set(), "eval_tags": set(), "mem": {}}
    for line in out.splitlines():
        if line.startswith("L "):
            v = line[2:].strip()
            if v.isdigit():
                st["launcher_slots"].add(int(v))
        elif line.startswith("E "):
            tok = line[2:].split()
            for i, t in enumerate(tok[:-1]):
                if t == "--gpu_ids":
                    st["eval_gpu_sets"].add(tok[i + 1])
                if t == "--tag":
                    st["eval_tags"].add(tok[i + 1])
        elif line.startswith("G "):
            a, b = [x.strip() for x in line[2:].split(",")]
            st["mem"][int(a)] = int(b)
    return st


def slot_free(st, slot):
    ids = [int(x) for x in GPUS[slot].split(",")]
    return (slot not in st["launcher_slots"] and GPUS[slot] not in st["eval_gpu_sets"]
            and all(st["mem"].get(i, 99999) < 2000 for i in ids))


def math_score(m, step):
    r = done_tags(m).get(f"{m}_global_step_{step}")
    return None if not r else r.get("math500", {}).get("avg@8")


def best_step(m):
    """Best step by math500 avg@8 over all 4 ckpts; None until all four are scored."""
    sc = {s: math_score(m, s) for s in ALL_STEPS}
    if any(v is None for v in sc.values()):
        return None
    return max(ALL_STEPS, key=lambda s: (sc[s], s))


def build_jobs():
    """Priority-ordered list of (job_id, method, model_path, tag, benchmarks, out_method)."""
    jobs = []
    # Baselines' math500 first (user request 10-29): needed to read any method's gain.
    for tag, path in BASELINE_MATH.items():
        jobs.append((f"base:{tag}", "vanilla_opd", path, tag, MATH, "vanilla_opd"))
    for s in STEPS_TODO:
        for m in METHODS:
            if launcher_reached_eval(m) and ckpt_ready(m, s):
                jobs.append((f"{m}:{s}", m, f"{run_dir(m)}/hf/global_step_{s}", f"{m}_global_step_{s}", MATH, m))
    for m in METHODS:
        b = best_step(m)
        if b is not None:
            jobs.append((f"{m}:ext{b}", m, f"{run_dir(m)}/hf/global_step_{b}", f"{m}_global_step_{b}_ext", EXT, m))
    for tag, path in BASELINES.items():
        jobs.append((f"base:{tag}", "vanilla_opd", path, tag, EXT, "vanilla_opd"))
    return [j for j in jobs if j[3] not in done_tags(j[5])]


def launch(ip, slot, job):
    _, m, model, tag, bench, outm = job
    out = f"{run_dir(outm)}/eval"
    logf = f"{run_dir(outm)}/logs/eval_{tag}.log"
    os.makedirs(out, exist_ok=True)
    cmd = (f"cd /root/EasyOPD && setsid nohup env PYTHONPATH=/root/EasyOPD VLLM_USE_V1=0 "
           f"VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=true {PY} "
           f"/root/EasyOPD/experiments/07_opd_repro/scripts/evaluate.py --model_path {model} --data_dir {DATA} "
           f"--output_dir {out} --tag {tag} --benchmarks {bench} --temperature 0.6 --top_p 0.95 --max_tokens 8192 "
           f"--max_prompt_len 1024 --dp_size 4 --gpu_ids {GPUS[slot]} > {logf} 2>&1 < /dev/null &")
    subprocess.run(["ssh", "-n", "-f", "-o", "BatchMode=yes", f"root@{ip}", cmd], timeout=40, check=True)
    log(f"launched {tag} [{bench}] on {ip} gpus {GPUS[slot]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    launched = {}   # tag -> (time, attempts)
    while True:
        jobs = build_jobs()
        if not jobs and all(best_step(m) is not None for m in METHODS):
            log("all jobs done"); break
        states = {ip: probe(ip) for ip in NODES}
        running_tags = set().union(*[s["eval_tags"] for s in states.values() if s])
        for ip in NODES:
            st = states[ip]
            if not st:
                continue
            for slot in (0, 1):
                if not slot_free(st, slot):
                    continue
                for job in jobs:
                    tag = job[3]
                    t, n = launched.get(tag, (0, 0))
                    if tag in running_tags or time.time() - t < 300 or n >= MAX_ATTEMPTS:
                        continue
                    if args.dry_run:
                        log(f"[dry] would launch {tag} on {ip} slot {slot}")
                    else:
                        launch(ip, slot, job)
                        launched[tag] = (time.time(), n + 1)
                    running_tags.add(tag)
                    break
        for tag, (t, n) in launched.items():
            if n >= MAX_ATTEMPTS and tag not in running_tags and tag in {j[3] for j in jobs}:
                log(f"WARN {tag} failed {n} times; see eval log")
        if args.dry_run:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
