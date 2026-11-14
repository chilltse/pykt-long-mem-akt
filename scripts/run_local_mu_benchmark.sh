#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/run_mu_benchmark.sh --mode offline --config configs/experiments/local_mu_probe.yaml --output output/kt_mc_mu_probe/local --candidate-limit 3 --student-limit 2 --selection-mc 4 --evaluation-mc 16 --reference-mc 32
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kt-mpl}" "${PYTHON_BIN:-/Users/tse/miniconda3/envs/pykt/bin/python}" scripts/report_local_mu.py output/kt_mc_mu_probe/local
