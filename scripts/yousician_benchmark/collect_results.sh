#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/collect_results.py" "$@"
