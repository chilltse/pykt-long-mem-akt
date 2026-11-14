#!/usr/bin/env bash
set -euo pipefail
source "$BENCHMARK_CLUSTER_ENV"
if [[ -n "${CONDA_SH:-}" ]]; then source "$CONDA_SH"; conda activate "$CONDA_ENV"; fi
cd "$BENCHMARK_CODE_ROOT"
export OMP_NUM_THREADS=${TRAIN_CPUS:-4}
python -c 'import torch,sys; print(sys.version, torch.__version__); assert torch.cuda.is_available(), "No visible CUDA GPU"; print(torch.cuda.get_device_name())'
python examples/benchmark_yousician_fmkc.py train --run_dir "$BENCHMARK_RUN_DIR" --task_id "$SLURM_ARRAY_TASK_ID" --require_cuda
