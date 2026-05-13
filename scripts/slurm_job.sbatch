#!/usr/bin/env bash
#SBATCH --job-name=project-template
#SBATCH --time=06:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

entrypoint="${1:-train}"
shift || true

case "${entrypoint}" in
  train)
    ./px run train "$@"
    ;;
  sample)
    ./px run sample "$@"
    ;;
  eval)
    ./px run eval "$@"
    ;;
  pipeline)
    ./px run pipeline "$@"
    ;;
  *)
    echo "unknown entrypoint: ${entrypoint}" >&2
    exit 2
    ;;
esac
