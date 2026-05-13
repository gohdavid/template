#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/submit

launcher="${1:-preempt_h200}"
shift || true

echo "Submitting training sweep with hydra/launcher=${launcher}"
nohup ./px run train -m "hydra/launcher=${launcher}" "$@" \
  >"logs/submit/train_${launcher}.log" 2>&1 < /dev/null &

echo "submission_pid=$!"
echo "submission_log=logs/submit/train_${launcher}.log"
