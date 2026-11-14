#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${BENCHMARK_CLUSTER_ENV:-$HERE/cluster.env}
if [[ ! -f "$ENV_FILE" ]]; then echo "Configure $ENV_FILE using cluster.env.example" >&2; exit 1; fi
set -a
source "$ENV_FILE"
set +a
: "${PROJECT_ROOT:?}" "${BENCHMARK_OUTPUT_ROOT:?}" "${YOUSICIAN_RAW_FILE:?}"
if [[ -n "${CONDA_SH:-}" ]]; then source "$CONDA_SH"; conda activate "$CONDA_ENV"; fi
export OMP_NUM_THREADS=${TRAIN_CPUS:-4}
export MPLCONFIGDIR=${TMPDIR:-/tmp}/yousician-mpl-${USER:-user}
