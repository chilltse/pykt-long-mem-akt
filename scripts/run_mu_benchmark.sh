#!/usr/bin/env bash
set -euo pipefail
PYKT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIANOKT_ROOT="${PIANOKT_ROOT:-$(dirname "$PYKT_ROOT")/../pianokt}"
PYTHON_BIN="${PYTHON_BIN:-/Users/tse/miniconda3/envs/pykt/bin/python}"
export PYTHONPATH="$PYKT_ROOT:$PIANOKT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kt-mpl}"
if [ "$#" -eq 0 ]; then
  set -- --mode mock --output "$PYKT_ROOT/output/kt_mc_mu_probe/smoke" --selection-mc 4 --evaluation-mc 8 --reference-mc 16
fi
exec "$PYTHON_BIN" -m pianokt_backend.experiments.cli "$@"
