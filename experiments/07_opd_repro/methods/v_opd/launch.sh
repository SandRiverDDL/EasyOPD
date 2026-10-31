#!/bin/bash
# 07_opd_repro / v_opd: manual single-run entry. Method switch and knobs live in
# ../../configs/methods/v_opd.yaml (the train scheduler reads the same file); shared protocol in ../../common.sh.
# Any variable can be overridden from the environment, e.g.
#   TOTAL_STEPS=20 SAVE_FREQ=10 EVAL_STEPS=last bash launch.sh
_D="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
_CFG=$("${ENV_DIR:-/root/envs/easyopd}/bin/python" "${_D}/../../scripts/expcfg.py" bash "${_D}/../../configs/methods/v_opd.yaml") || exit 1
eval "${_CFG}"
source "${_D}/../../common.sh"
