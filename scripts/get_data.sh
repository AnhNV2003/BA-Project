#!/usr/bin/env bash
# Fetch the raw PaySim CSV and rebuild every processed/model artifact.
#
# Usage:
#   ./scripts/get_data.sh            # download via Kaggle CLI (if configured) + full rebuild
#   ./scripts/get_data.sh --no-download   # skip download, assume CSV already in data/raw/
#   ./scripts/get_data.sh --sample 0.15   # rebuild on a stratified sample instead of full data
#
# Requires (for download): `pip install kaggle` + ~/.kaggle/kaggle.json API token
# (https://www.kaggle.com/docs/api -> Account -> Create New API Token).
# Without that, download the CSV by hand from:
#   https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
# and place it at data/raw/PS_20174392719_1491204439457_log.csv, then re-run with
# --no-download.
set -euo pipefail

cd "$(dirname "$0")/.."

CSV_NAME="PS_20174392719_1491204439457_log.csv"
CSV_PATH="data/raw/${CSV_NAME}"
KAGGLE_DATASET="rupakroy/online-payments-fraud-detection-dataset"

DOWNLOAD=1
BUILD_MODE="--full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-download) DOWNLOAD=0; shift ;;
        --sample) BUILD_MODE="--frac ${2}"; shift 2 ;;
        --full) BUILD_MODE="--full"; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p data/raw data/synthetic data/processed models

if [[ -f "$CSV_PATH" ]]; then
    echo "[get_data] found existing $CSV_PATH, skipping download"
elif [[ "$DOWNLOAD" -eq 1 ]]; then
    if ! command -v kaggle >/dev/null 2>&1; then
        cat >&2 <<EOF
[get_data] 'kaggle' CLI not found. Install it and configure your API token:
    pip install kaggle
    # place kaggle.json (from kaggle.com/settings) at ~/.kaggle/kaggle.json
Then re-run this script, or download the CSV manually from:
    https://www.kaggle.com/datasets/${KAGGLE_DATASET}
and place it at ${CSV_PATH}, then re-run with --no-download.
EOF
        exit 1
    fi
    echo "[get_data] downloading ${KAGGLE_DATASET} via Kaggle CLI..."
    kaggle datasets download -d "$KAGGLE_DATASET" -p data/raw --unzip
    if [[ ! -f "$CSV_PATH" ]]; then
        found=$(find data/raw -maxdepth 1 -name "*.csv" | head -n1 || true)
        if [[ -n "$found" && "$found" != "$CSV_PATH" ]]; then
            mv "$found" "$CSV_PATH"
        fi
    fi
else
    echo "[get_data] --no-download set but $CSV_PATH is missing." >&2
    echo "[get_data] Download it manually first (see script header)." >&2
    exit 1
fi

test -f "$CSV_PATH" || { echo "[get_data] ERROR: $CSV_PATH still missing after download step." >&2; exit 1; }
echo "[get_data] raw CSV ready: $CSV_PATH ($(du -h "$CSV_PATH" | cut -f1))"

export PYTHONPATH="src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "[get_data] === M1 build_dataset ($BUILD_MODE) ==="
python src/build_dataset.py $BUILD_MODE

echo "[get_data] === M3 cleaning ==="
python src/cleaning.py

echo "[get_data] === M4 features ==="
python src/features.py

echo "[get_data] === M5 train_validate ==="
python src/train_validate.py

echo "[get_data] done. Artifacts:"
echo "  data/processed/transactions_context.parquet"
echo "  data/processed/transactions_clean.parquet"
echo "  models/feature_transformer.joblib"
echo "  models/fraud_model.joblib"
