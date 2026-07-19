#!/usr/bin/env bash
# PaySim fraud pipeline runner.
#
#   ./train.sh                 # train only, on the EXISTING processed parquet
#   ./train.sh full            # rebuild the WHOLE pipeline on all 6.36M rows, then train
#   ./train.sh sample          # rebuild on the default 15% stratified sample, then train
#   ./train.sh sample 0.3      # rebuild on a custom stratified fraction, then train
#
# For the full run (long + memory-heavy) prefer:
#   nohup ./train.sh full > train.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

MODE="${1:-train}"      # train | full | sample
FRAC="${2:-0.15}"       # only used in sample mode

rebuild() {
    echo "=== M1 build_dataset ($1) ==="
    python src/build_dataset.py $2
    echo "=== M3 cleaning ==="
    python src/cleaning.py
    echo "=== M4 features ==="
    python src/features.py
}

case "$MODE" in
    full)
        rebuild "full 6.36M rows" "--full"
        ;;
    sample)
        rebuild "${FRAC} stratified sample" "--frac ${FRAC}"
        ;;
    train)
        echo "=== training on existing data/processed/transactions_clean.parquet ==="
        ;;
    *)
        echo "unknown mode: $MODE (use: train | full | sample [frac])" >&2
        exit 1
        ;;
esac

echo "=== M5 train_validate ==="
python src/train_validate.py
