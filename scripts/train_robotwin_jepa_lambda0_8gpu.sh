#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/workspace/Tao/motus_LDJEPA"
PYTHON_BIN="/mnt/workspace/Tao/.conda/envs/motus-robodojo/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Training interpreter is unavailable: $PYTHON_BIN" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config configs/robotwin_jepa_lambda0.yaml \
    --run_name robotwin_jepa_lambda0 \
    --report_to tensorboard