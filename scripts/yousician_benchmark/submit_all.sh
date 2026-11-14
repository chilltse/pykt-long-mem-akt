#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
args=(); sizes=(60 120 240); models=(akt akt_fmkc); folds=(0); seeds=(3407); full=0; dry=0
while (($#)); do
  case "$1" in
    --dry-run) dry=1; shift;;
    --include-full) full=1; args+=(--include_full); shift;;
    --train-student-sizes|--models|--fold|--folds|--seed|--seeds)
      key=$1; shift; values=()
      while (($#)) && [[ "$1" != --* ]]; do values+=("$1"); shift; done
      ((${#values[@]})) || { echo "Missing values for $key" >&2; exit 1; }
      case "$key" in
        --train-student-sizes) sizes=("${values[@]}"); args+=(--train_student_sizes "${values[@]}");;
        --models) models=("${values[@]}"); args+=(--models "${values[@]}");;
        --fold|--folds) folds=("${values[@]}"); args+=(--folds "${values[@]}");;
        --seed|--seeds) seeds=("${values[@]}"); args+=(--seeds "${values[@]}");;
      esac;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done
[[ -f "$YOUSICIAN_RAW_FILE" ]] || { echo "Dataset missing" >&2; exit 1; }
count=$(( (${#sizes[@]} + full) * ${#models[@]} * ${#folds[@]} * ${#seeds[@]} ))
run_id="run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
export BENCHMARK_RUN_DIR="$BENCHMARK_OUTPUT_ROOT/$run_id"
export BENCHMARK_CODE_ROOT="$BENCHMARK_OUTPUT_ROOT/${run_id}_code"
export BENCHMARK_CLUSTER_ENV="$ENV_FILE"
logs="$BENCHMARK_OUTPUT_ROOT/${run_id}_logs"
shared=(); [[ -z "${SLURM_ACCOUNT:-}" ]] || shared+=(--account="$SLURM_ACCOUNT")
[[ -z "${SLURM_QOS:-}" ]] || shared+=(--qos="$SLURM_QOS")
cpu=(); gpu=(); [[ -z "${CPU_PARTITION:-}" ]] || cpu+=(--partition="$CPU_PARTITION")
[[ -z "${GPU_PARTITION:-}" ]] || gpu+=(--partition="$GPU_PARTITION")
prepare=(sbatch --parsable ${shared[@]+"${shared[@]}"} ${cpu[@]+"${cpu[@]}"} --cpus-per-task="$PREPARE_CPUS" --mem="$PREPARE_MEMORY" --time="$PREPARE_TIME" --output="$logs/prepare_%j.out" --error="$logs/prepare_%j.err")
train=(sbatch --parsable ${shared[@]+"${shared[@]}"} ${gpu[@]+"${gpu[@]}"} "${GPU_RESOURCE:---gres=gpu:1}" --cpus-per-task="$TRAIN_CPUS" --mem="$TRAIN_MEMORY" --time="$TRAIN_TIME" --array="0-$((count-1))%$MAX_CONCURRENT_GPU_JOBS" --output="$logs/train_%A_%a.out" --error="$logs/train_%A_%a.err")
analyze=(sbatch --parsable ${shared[@]+"${shared[@]}"} ${cpu[@]+"${cpu[@]}"} --cpus-per-task="$ANALYZE_CPUS" --mem="$ANALYZE_MEMORY" --time="$ANALYZE_TIME" --output="$logs/analyze_%j.out" --error="$logs/analyze_%j.err")
echo "Run: $BENCHMARK_RUN_DIR; tasks=$count; sizes=${sizes[*]}; full=$full; models=${models[*]}; folds=${folds[*]}; seeds=${seeds[*]}"
if ((dry)); then
  printf '%q ' "${prepare[@]}" "$HERE/prepare.sbatch" ${args[@]+"${args[@]}"}; echo
  printf '%q ' "${train[@]}" '--dependency=afterok:PREPARE_JOB' "$HERE/train_array.sbatch"; echo
  printf '%q ' "${analyze[@]}" '--dependency=afterany:TRAIN_JOB' "$HERE/analyze.sbatch"; echo
  exit 0
fi
command -v sbatch >/dev/null
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no)" ]] || { echo 'Commit tracked changes before submission' >&2; exit 1; }
mkdir -p "$logs"
git -C "$PROJECT_ROOT" worktree add --detach "$BENCHMARK_CODE_ROOT" HEAD
# Jobs use an immutable detached checkout; task startup also verifies its fingerprint.
prepare_id=$("${prepare[@]}" "$BENCHMARK_CODE_ROOT/scripts/yousician_benchmark/prepare.sbatch" ${args[@]+"${args[@]}"}); prepare_id=${prepare_id%%;*}
[[ "$prepare_id" =~ ^[0-9]+$ ]]
train_id=$("${train[@]}" --dependency="afterok:$prepare_id" "$BENCHMARK_CODE_ROOT/scripts/yousician_benchmark/train_array.sbatch"); train_id=${train_id%%;*}
[[ "$train_id" =~ ^[0-9]+$ ]]
analyze_id=$("${analyze[@]}" --dependency="afterany:$train_id" "$BENCHMARK_CODE_ROOT/scripts/yousician_benchmark/analyze.sbatch"); analyze_id=${analyze_id%%;*}
[[ "$analyze_id" =~ ^[0-9]+$ ]]
printf 'prepare=%s\ntrain=%s\nanalyze=%s\nrun_dir=%s\ncode_root=%s\n' "$prepare_id" "$train_id" "$analyze_id" "$BENCHMARK_RUN_DIR" "$BENCHMARK_CODE_ROOT" > "$logs/submission.txt"
echo "Monitor: squeue -j $prepare_id,$train_id,$analyze_id"
