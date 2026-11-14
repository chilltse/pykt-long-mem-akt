#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
python -c 'import torch,numpy,sklearn,pytest; print("PyTorch",torch.__version__)'
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/yousician-mpl
for file in scripts/yousician_benchmark/*.sh scripts/yousician_benchmark/*.sbatch; do bash -n "$file"; done
python -m pytest -q tests/test_benchmark_prepare.py tests/test_benchmark_metrics.py tests/test_vector_kc.py tests/test_yousician_akt.py tests/test_akt_fmkc_incremental.py tests/test_fmkc_serving.py
OUT=$(mktemp -d "${TMPDIR:-/tmp}/yousician-local.XXXXXX")
python examples/benchmark_yousician_fmkc.py smoke --run_dir "$OUT/synthetic"
python examples/benchmark_yousician_fmkc.py evaluate --run_dir "$OUT/synthetic" --task_id 0
python examples/benchmark_yousician_fmkc.py analyze --run_dir "$OUT/synthetic" --bootstrap_replicates 10
PYTHONPATH=. python tests/test_benchmark_export.py "$OUT/synthetic"
python examples/benchmark_yousician_fmkc.py status --run_dir "$OUT/synthetic"
python - "$OUT/cluster.env" "$ROOT" "$OUT" <<'PY'
import sys,shlex
from pathlib import Path
text=Path('scripts/yousician_benchmark/cluster.env.example').read_text()
text+='\nPROJECT_ROOT='+shlex.quote(sys.argv[2])+'\nBENCHMARK_OUTPUT_ROOT='+shlex.quote(sys.argv[3])+'\nYOUSICIAN_RAW_FILE='+shlex.quote(str(Path(sys.argv[3])/'synthetic/dataset_audit.json'))+'\n'
Path(sys.argv[1]).write_text(text)
PY
BENCHMARK_CLUSTER_ENV="$OUT/cluster.env" bash scripts/yousician_benchmark/submit_all.sh --dry-run --train-student-sizes 60 120 240 --include-full
bash scripts/yousician_benchmark/collect_results.sh --run-dir "$OUT/synthetic" --mode summary
echo "Local synthetic artifacts: $OUT/synthetic"
