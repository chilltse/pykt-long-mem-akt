#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
[[ "${1:-}" == --run-dir && -n "${2:-}" ]] || { echo 'Usage: retry_failed.sh --run-dir DIR' >&2; exit 1; }
export BENCHMARK_RUN_DIR=$(cd "$2" && pwd)
export BENCHMARK_CODE_ROOT="${BENCHMARK_RUN_DIR}_code"
export BENCHMARK_CLUSTER_ENV="$ENV_FILE"
[[ -d "$BENCHMARK_CODE_ROOT" ]] || { echo 'Immutable code checkout not found' >&2; exit 1; }
cd "$BENCHMARK_CODE_ROOT"
ids=$(python - "$BENCHMARK_RUN_DIR" <<'PY'
import sys
from pathlib import Path
from pykt.benchmark.runtime import read,complete
print(','.join(str(t['task_id']) for t in read(Path(sys.argv[1])/'task_manifest.json') if not complete(t)))
PY
)
[[ -n "$ids" ]] || { echo 'All tasks complete'; exit 0; }
logs="${BENCHMARK_RUN_DIR}_logs";mkdir -p "$logs"
options=(); [[ -z "${SLURM_ACCOUNT:-}" ]] || options+=(--account="$SLURM_ACCOUNT")
[[ -z "${GPU_PARTITION:-}" ]] || options+=(--partition="$GPU_PARTITION")
[[ -z "${SLURM_QOS:-}" ]] || options+=(--qos="$SLURM_QOS")
sbatch --parsable ${options[@]+"${options[@]}"} "${GPU_RESOURCE:---gres=gpu:1}" --array="$ids%$MAX_CONCURRENT_GPU_JOBS" --cpus-per-task="$TRAIN_CPUS" --mem="$TRAIN_MEMORY" --time="$TRAIN_TIME" --output="$logs/retry_%A_%a.out" --error="$logs/retry_%A_%a.err" scripts/yousician_benchmark/train_array.sbatch
# The train command reuses intact checkpoints for evaluation; interrupted training restarts.
echo "After completion: python examples/benchmark_yousician_fmkc.py analyze --run_dir '$BENCHMARK_RUN_DIR'"
