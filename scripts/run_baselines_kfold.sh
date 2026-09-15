#!/bin/bash
# shellcheck disable=SC2102
# Standard sklearn baselines (SVM + Elastic Net) with stratified 5-fold CV.
# Same split protocol as scripts/rerun_best_kfold.sh (GNNs).
#
# W&B project: ogbench_kfold
# Aggregate across folds with dataset.split_params.data_seed (0..4).
#
# Launch: bash scripts/run_baselines_kfold.sh
# Optional: MAX_PARALLEL=8 N_THREADS=4 bash scripts/run_baselines_kfold.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/lcornelis/ogbench}"
WANDB_PROJECT="${WANDB_PROJECT:-ogbench_kfold}"
WANDB_ENTITY="${WANDB_ENTITY:-bioshape-lab}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
N_THREADS="${N_THREADS:-4}"
K="${K:-5}"

DATASETS=(
  addneuromed
  motrpac
  parkinsons
  brca
)

running=0

launch() {
  echo "[threads=${N_THREADS}] $*"
  OMP_NUM_THREADS="${N_THREADS}" \
    MKL_NUM_THREADS="${N_THREADS}" \
    OPENBLAS_NUM_THREADS="${N_THREADS}" \
    NUMEXPR_NUM_THREADS="${N_THREADS}" \
    "$@" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n || true
    running=$((running - 1))
  fi
}

echo "========================================="
echo "K-fold standard baselines (SVM + Elastic Net)"
echo "  datasets: ${DATASETS[*]}"
echo "  k=${K}  project=${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "  jobs: $((${#DATASETS[@]} * K))  (each job runs both baselines)"
echo "========================================="

for dataset in "${DATASETS[@]}"; do
  for fold in $(seq 0 $((K - 1))); do
    launch python ogbench/baseline.py \
      dataset="${dataset}" \
      baseline_filter=standard \
      seed=42 \
      dataset.split_params.split_type=k-fold \
      dataset.split_params.k="${K}" \
      dataset.split_params.data_seed="${fold}" \
      tags="[baseline,sklearn,kfold,${dataset},fold${fold}]" \
      logger.wandb.entity="${WANDB_ENTITY}" \
      logger.wandb.project="${WANDB_PROJECT}" \
      paths.root_dir="${ROOT_DIR}"
  done
done

wait || true

echo ""
echo "All k-fold baseline jobs finished."
echo "Expected W&B runs: $((${#DATASETS[@]} * K * 2))  (datasets × folds × {svm, elastic_net})"
echo "Project: ${WANDB_ENTITY}/${WANDB_PROJECT}"
