#!/usr/bin/env python3
"""Experiment yaml for 07_opd_repro: `extends` + merge -> bash variables for common.sh.

New code (no upstream reference). common.sh stays the general base (its defaults are not
copied into yaml); a method yaml (configs/methods/<m>.yaml) holds the method switch and the
method-intrinsic knobs; an experiment yaml extends protocol / method yamls and only lists what
this run changes.

Schema (top-level keys):
  extends: str | list[str]   paths relative to this file, merged in order, this file on top
  method:  vanilla_opd       required after merge -> METHOD
  name:    full              run name; the scheduler makes the tag <MMDD_HHMM>_<name>
  gpus:    4                 4 (one GPU half of a node) or 8 (whole node); scheduler only
  env:     {VAR: value}      common.sh variables (TOTAL_STEPS, TRAIN_PARQUET, ...)
  args:    {hydra.key: v}    extra Hydra overrides, rendered into OPD_ARGS as key=value
  note:    free text         copied to the run's resolved.yaml, not used
Merge rule: dicts merge by key (recursively); scalars and lists are replaced; a null value
deletes the inherited key. `args` is a dict (not a list) so a later layer can override or
delete one Hydra key without restating the others.
Precedence (low -> high): common.sh defaults < extends (in order) < this file < variables
already set in the calling shell (so `TOTAL_STEPS=2 bash methods/x/launch.sh` still works).

CLI:
  expcfg.py show <yaml>   print the resolved yaml
  expcfg.py bash <yaml>   print bash to eval before sourcing common.sh
"""
import os
import re
import shlex
import sys

import yaml

TOP_KEYS = {"extends", "method", "name", "gpus", "env", "args", "note"}
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path, _seen=()):
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"extends cycle: {' -> '.join(_seen + (path,))}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    bad = set(raw) - TOP_KEYS
    if bad:
        raise ValueError(f"{path}: unknown top-level keys {sorted(bad)} (allowed {sorted(TOP_KEYS)})")
    parents = raw.pop("extends", None) or []
    if isinstance(parents, str):
        parents = [parents]
    cfg = {}
    for p in parents:
        cfg = _merge(cfg, load(os.path.join(os.path.dirname(path), p), _seen + (path,)))
    return _merge(cfg, raw)


def validate(cfg, path=""):
    if not cfg.get("method"):
        raise ValueError(f"{path}: 'method' is not set by the file or its extends")
    if cfg.get("gpus") not in (None, 4, 8):
        raise ValueError(f"{path}: gpus must be 4 or 8, got {cfg.get('gpus')!r}")
    for k, v in (cfg.get("env") or {}).items():
        if not _ENV_RE.match(k):
            raise ValueError(f"{path}: env key {k!r} is not an UPPER_CASE shell variable")
        if isinstance(v, (dict, list)):
            raise ValueError(f"{path}: env.{k} must be a scalar")
    for k, v in (cfg.get("args") or {}).items():
        if isinstance(v, dict):
            raise ValueError(f"{path}: args.{k} must be a scalar or list (write nested keys dotted)")
    return cfg


def _hydra_value(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, list):
        return "[" + ",".join(_hydra_value(x) for x in v) + "]"
    return str(v)


def _env_value(v):
    return _hydra_value(v) if isinstance(v, bool) else str(v)


def hydra_args(cfg):
    return [f"{k}={_hydra_value(v)}" for k, v in (cfg.get("args") or {}).items()]


def to_bash(cfg):
    lines = [f"METHOD={shlex.quote(cfg['method'])}"]
    for k, v in (cfg.get("env") or {}).items():
        # Caller's shell wins: lets manual runs override any yaml value from the environment.
        lines.append(f'[ -n "${{{k}+x}}" ] || export {k}={shlex.quote(_env_value(v))}')
    lines.append("OPD_ARGS=(" + " ".join(shlex.quote(a) for a in hydra_args(cfg)) + ")")
    return "\n".join(lines) + "\n"


def resolve(path):
    return validate(load(path), path)


def main(argv):
    if len(argv) != 3 or argv[1] not in ("show", "bash"):
        sys.exit(__doc__)
    cfg = resolve(argv[2])
    if argv[1] == "show":
        sys.stdout.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    else:
        sys.stdout.write(to_bash(cfg))


if __name__ == "__main__":
    main(sys.argv)
