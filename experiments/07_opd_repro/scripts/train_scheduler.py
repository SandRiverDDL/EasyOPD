#!/usr/bin/env python3
"""Training queue for 07_opd_repro: submit experiment yamls, launch them on free GPU halves.

New code; the node probe / slot logic and the ssh + setsid launch are reused from
scripts/eval_scheduler.py and scripts/run_ablation.sh. Runs on the driver machine (NFS visible,
ssh to the nodes). Training <-> eval coordination is not handled here yet.

  train_scheduler.py submit <exp.yaml>...   resolve (scripts/expcfg.py) and append to the queue
  train_scheduler.py status                 pending / running / done / failed, with step progress
  train_scheduler.py cancel <name>          drop a *pending* entry (never touches a started run)
  train_scheduler.py run [--dry_run]        loop: launch pending entries (FIFO, first fit) onto
                                            free slots; idles while the queue is empty
Queue (NFS): runs/train/07_opd_repro/_queue/{pending,launched}/<seq>_<name>.yaml.
A slot is free when no 07 launcher (bash with SLOT/N_GPUS in its environ) or evaluate.py holds
it, every GPU in it uses < 2000 MiB, and no entry was launched onto it in the last GRACE_S
seconds (covers Ray start-up before GPU memory shows). Only NODES are touched; nothing is killed.
Launch: rsync the NFS repo to the node's /root/EasyOPD, hard-link snapshot it to
/root/easyopd_code/<tag> (EASYOPD_ROOT; later rsyncs do not change a running job's code), write
<run_dir>/{resolved.yaml,launch.sh,code_version.txt}, start `setsid nohup bash launch.sh`.
"""
import argparse
import fcntl
import os
import re
import shlex
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expcfg  # noqa: E402
import run_table  # noqa: E402

NLU = "/mnt/NLU_disk17/chenhaoyu02"
REPO = f"{NLU}/EasyOPD"
RUNS = f"{NLU}/runs/train/07_opd_repro"
QUEUE = f"{RUNS}/_queue"
NODES = ["10.206.120.80", "10.206.114.79", "10.206.120.12"]  # the only machines this script touches
HALVES = {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)}
GRACE_S = 900
TABLE_EVERY_S = 600         # rebuild RUNS/RUNS.md (scripts/run_table.py) this often
MIN_FREE_GB = 300           # FSDP ckpts (weights + optimizer) land on the node's local disk
OWNED_ENV = {"SLOT", "N_GPUS", "GPU_IDS", "RUN_TAG", "RUN_DIR", "EASYOPD_ROOT"}  # set by the scheduler
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

PROBE = r"""
ps -eo pid=,args= | awk '$2=="bash" && /07_opd_repro/ {print $1}' | while read p; do
  tr '\0' '\n' </proc/$p/environ 2>/dev/null | awk -F= '$1=="SLOT"{s=$2} $1=="N_GPUS"{n=$2} $1=="RUN_TAG"{t=$2}
    END{if (s != "") print "L", s, (n == "" ? 8 : n), t}'; done | sort -u
ps -eo args= | grep -E 'evaluate.py --model_path' | grep -v -e '--worker' -e grep | sed 's/^/E /'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/^/G /'
df -BG --output=avail /root | tail -1 | tr -dc 0-9 | sed 's/^/D /'; echo
"""


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def _entries(state):
    d = f"{QUEUE}/{state}"
    os.makedirs(d, exist_ok=True)
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".yaml"):
            with open(f"{d}/{f}") as fh:
                out.append((f"{d}/{f}", yaml.safe_load(fh)))
    return out


def _write_atomic(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


def cmd_submit(paths):
    for p in paths:
        cfg = expcfg.resolve(p)
        if not cfg.get("name"):
            sys.exit(f"{p}: 'name' is required (tag = <MMDD_HHMM>_<name>)")
        cfg.setdefault("gpus", 4)
        clash = OWNED_ENV & set(cfg.get("env") or {})
        if clash:
            sys.exit(f"{p}: env {sorted(clash)} is set by the scheduler; use 'gpus' instead")
        seq = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 10**6:06d}"
        entry = {"source": os.path.abspath(p), "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "cfg": cfg}
        _write_atomic(f"{QUEUE}/pending/{seq}_{cfg['name']}.yaml", entry)
        log(f"queued {cfg['name']} ({cfg['method']}, {cfg['gpus']} GPUs) from {p}")


def cmd_cancel(name):
    for path, e in _entries("pending"):
        if e["cfg"]["name"] == name:
            os.remove(path)
            log(f"cancelled pending {name}")
            return
    sys.exit(f"no pending entry named {name} (started runs are never cancelled here)")


def probe(ip):
    """{'held': set of GPU ids held by launchers/evals, 'tags': running RUN_TAGs, 'mem': {gpu: MiB}, 'disk_gb'}."""
    try:
        out = subprocess.run(SSH + [f"root@{ip}", PROBE], capture_output=True, text=True, timeout=40).stdout
    except subprocess.TimeoutExpired:
        return None
    st = {"held": set(), "tags": set(), "mem": {}, "disk_gb": None}
    for line in out.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "L" and len(tok) >= 3 and tok[1].isdigit() and tok[2].isdigit():
            s, n = int(tok[1]), int(tok[2])
            st["held"].update(range(s * n, s * n + n))
            if len(tok) >= 4:
                st["tags"].add(tok[3])
        elif tok[0] == "E":
            for i, t in enumerate(tok[:-1]):
                if t == "--gpu_ids":
                    st["held"].update(int(x) for x in tok[i + 1].split(","))
        elif tok[0] == "G":
            a, b = line[2:].split(",")
            st["mem"][int(a)] = int(b)
        elif tok[0] == "D" and len(tok) == 2:
            st["disk_gb"] = int(tok[1])
    return st if st["mem"] else None


def free_halves(st, reserved):
    """GPU halves (0 = GPUs 0-3, 1 = GPUs 4-7) with no launcher/eval, < 2000 MiB used and no fresh launch."""
    def ok(gpus):
        return all(g not in st["held"] and g not in reserved and st["mem"].get(g, 99999) < 2000 for g in gpus)
    return [s for s, g in HALVES.items() if ok(g)]


def place(n_gpus, halves):
    """(SLOT, halves used) for a job, or None. 8 GPUs need the whole node and run as SLOT=0 N_GPUS=8."""
    if n_gpus == 8:
        return (0, [0, 1]) if halves == [0, 1] else None
    return (halves[0], [halves[0]]) if halves else None


def _tail(path, nbytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - nbytes))
            return f.read().decode(errors="ignore")
    except OSError:
        return ""


def run_state(e, running_tags):
    """('running'|'starting'|'done'|'failed', last step) for a launched entry."""
    rd, tag, method = e["run_dir"], e["tag"], e["cfg"]["method"]
    steps = re.findall(r"step:(\d+) ", _tail(f"{rd}/logs/train.log"))
    last = int(steps[-1]) if steps else 0
    launcher = _tail(f"{rd}/logs/launcher.log", 20_000)
    if f"===== {method} done =====" in launcher:
        return "done", last
    if "[FATAL]" in launcher and tag not in running_tags:
        return "failed", last
    if tag in running_tags:
        return "running", last
    if time.time() - e["launched_ts"] < GRACE_S:
        return "starting", last
    return "failed", last


def _code_version():
    def git(*a):
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=120).stdout
    # Untracked files (e.g. experiments/ before its first commit) are listed but not captured.
    return f"commit {git('rev-parse', 'HEAD').strip()}\n{git('status', '--short')}\n{git('diff')}"


def launch(ip, slot, path, e, dry_run):
    cfg = e["cfg"]
    n = cfg["gpus"]
    tag = time.strftime("%m%d_%H%M") + "_" + cfg["name"]
    # Run dir = RUNS/<batch>/<tag>, batch = the configs/exps/<batch>/ dir of the source yaml (10-31; runs
    # launched before that stay under RUNS/<method>/). A yaml outside configs/exps/ falls back to the method.
    grp = os.path.basename(os.path.dirname(e["source"])) if "/configs/exps/" in e["source"] else cfg["method"]
    rd = f"{RUNS}/{grp}/{tag}"
    if os.path.exists(rd):
        tag += time.strftime("%S")
        rd = f"{RUNS}/{grp}/{tag}"
    code = f"/root/easyopd_code/{tag}"
    if dry_run:
        log(f"[dry] would launch {cfg['name']} -> {ip} slot {slot} ({n} GPUs) as {rd}")
        return True
    os.makedirs(f"{rd}/logs")
    _write_atomic(f"{rd}/resolved.yaml", {"source": e["source"], "node": ip, "slot": slot, "tag": tag, "cfg": cfg})
    with open(f"{rd}/code_version.txt", "w") as f:
        f.write(_code_version())
    with open(f"{rd}/launch.sh", "w") as f:
        f.write(f"#!/bin/bash\n# Generated by train_scheduler.py from {e['source']} ({time.strftime('%F %T')}).\n"
                f"# Rerun by hand on {ip}: SLOT={slot} N_GPUS={n} RUN_TAG={tag} RUN_DIR={rd} bash {rd}/launch.sh\n"
                f"export EASYOPD_ROOT={code}\n{expcfg.to_bash(cfg)}"
                f'source "${{EASYOPD_ROOT}}/experiments/07_opd_repro/common.sh"\n')
    env = f"SLOT={slot} N_GPUS={n} RUN_TAG={shlex.quote(tag)} RUN_DIR={shlex.quote(rd)}"
    remote = (f"set -e; mkdir -p /root/EasyOPD /root/easyopd_code; "
              f"rsync -a --exclude .git --exclude __pycache__ {REPO}/ /root/EasyOPD/; "
              f"rsync -a --link-dest=/root/EasyOPD /root/EasyOPD/ {code}/; cd /tmp; "
              f"env {env} setsid nohup bash {rd}/launch.sh > {rd}/logs/launcher.log 2>&1 < /dev/null &")
    r = subprocess.run(SSH + [f"root@{ip}", remote], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        log(f"ERROR launching {cfg['name']} on {ip}: rc={r.returncode} {r.stderr.strip()[-300:]}")
        return False
    e.update(tag=tag, run_dir=rd, node=ip, slot=slot, launched=time.strftime("%Y-%m-%d %H:%M:%S"), launched_ts=time.time())
    _write_atomic(f"{QUEUE}/launched/{os.path.basename(path)}", e)
    os.remove(path)
    log(f"launched {cfg['name']} -> {ip} slot {slot} ({n} GPUs): {rd}/logs/launcher.log")
    return True


def _probe_all():
    states = {ip: probe(ip) for ip in NODES}
    for ip, st in states.items():
        if st is None:
            log(f"WARN probe failed on {ip}; skipping it this round")
    running = set().union(*[st["tags"] for st in states.values() if st])
    return states, running


def cmd_status():
    states, running = _probe_all()
    rows = [("state", "name", "method", "gpus", "node/slot", "step", "tag / source")]
    for _, e in _entries("pending"):
        c = e["cfg"]
        rows.append(("pending", c["name"], c["method"], str(c["gpus"]), "-", "-", e["source"]))
    for _, e in _entries("launched"):
        c = e["cfg"]
        s, last = run_state(e, running)
        total = (c.get("env") or {}).get("TOTAL_STEPS", "?")
        rows.append((s, c["name"], c["method"], str(c["gpus"]), f"{e['node']}/{e['slot']}", f"{last}/{total}", e["tag"]))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(x.ljust(w[i]) for i, x in enumerate(r)).rstrip())
    for ip, st in states.items():
        if st:
            busy = sorted(st["held"] | {g for g, m in st["mem"].items() if m >= 2000})
            print(f"{ip}: busy GPUs {busy or '-'}, local disk {st['disk_gb']} GB free")


def schedule_once(dry_run):
    pending = _entries("pending")
    if not pending:
        return
    states, running = _probe_all()
    reserved = {ip: set() for ip in NODES}  # GPUs of entries launched < GRACE_S ago that are not visible yet
    for _, e in _entries("launched"):
        if e["node"] in reserved and run_state(e, running)[0] == "starting":
            n = e["cfg"]["gpus"]
            reserved[e["node"]].update(range(e["slot"] * n, e["slot"] * n + n))
    free = {ip: free_halves(st, reserved[ip]) for ip, st in states.items()
            if st and (st["disk_gb"] or 0) >= MIN_FREE_GB}
    for ip, st in states.items():
        if st and (st["disk_gb"] or 0) < MIN_FREE_GB:
            log(f"WARN {ip}: only {st['disk_gb']} GB local disk free (< {MIN_FREE_GB}); not scheduling there")
    for path, e in pending:  # FIFO, first fit: a 4-GPU entry may pass an 8-GPU one waiting for a whole node
        for ip in NODES:
            p = place(e["cfg"]["gpus"], free.get(ip, []))
            if p is None:
                continue
            if launch(ip, p[0], path, e, dry_run):
                free[ip] = [h for h in free[ip] if h not in p[1]]
            break


def cmd_run(interval, dry_run):
    os.makedirs(QUEUE, exist_ok=True)
    lock = open("/tmp/opd07_train_scheduler.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another train_scheduler.py run is active on this machine (/tmp/opd07_train_scheduler.lock)")
    lock.write(str(os.getpid()))
    lock.flush()
    log(f"scheduler pid {os.getpid()}, nodes {NODES}, queue {QUEUE}")
    last_table = 0
    while True:
        schedule_once(dry_run)
        if dry_run:
            return
        if time.time() - last_table > TABLE_EVERY_S:  # RUNS.md also picks up eval_scheduler's results
            last_table = time.time()
            try:
                run_table.build()
            except Exception as ex:  # a broken table must never stop scheduling
                log(f"WARN run_table.build failed: {ex!r}")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("submit").add_argument("yamls", nargs="+")
    sub.add_parser("status")
    sub.add_parser("cancel").add_argument("name")
    r = sub.add_parser("run")
    r.add_argument("--interval", type=int, default=60)
    r.add_argument("--dry_run", action="store_true", help="one round, print what would launch, change nothing")
    a = ap.parse_args()
    if a.cmd == "submit":
        cmd_submit(a.yamls)
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd == "cancel":
        cmd_cancel(a.name)
    else:
        cmd_run(a.interval, a.dry_run)


if __name__ == "__main__":
    main()
