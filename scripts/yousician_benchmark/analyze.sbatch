#!/usr/bin/env bash
set -euo pipefail
source "$BENCHMARK_CLUSTER_ENV"
if [[ -n "${CONDA_SH:-}" ]]; then source "$CONDA_SH"; conda activate "$CONDA_ENV"; fi
cd "$BENCHMARK_CODE_ROOT"
python examples/benchmark_yousician_fmkc.py analyze --run_dir "$BENCHMARK_RUN_DIR"
