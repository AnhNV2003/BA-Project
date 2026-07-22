# Data Setup

Heavy files are **not** committed to git (raw CSV ~471MB, processed parquet
~550-590MB each, model bundles). Only code + small docs/figures live in the
repo. Rebuild everything locally with one of the options below.

## Option A — one-shot script (recommended)

```bash
pip install -r requirements.txt
./scripts/get_data.sh
```

This downloads the PaySim CSV via the Kaggle CLI (if configured) and runs the
full pipeline end-to-end on **all 6,362,620 rows**: M1 build → M3 clean → M4
features → M5 train. Takes a few minutes; produces every artifact listed
below.

**Kaggle CLI setup** (one-time, needed for automatic download):
```bash
pip install kaggle
# Kaggle -> Account -> "Create New API Token" -> downloads kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

No Kaggle account/token? Download the CSV by hand instead:
1. Open https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
2. Download and unzip; place `PS_20174392719_1491204439457_log.csv` into `data/raw/`.
3. Run `./scripts/get_data.sh --no-download`.

**Faster local iteration** (15% stratified sample instead of full data):
```bash
./scripts/get_data.sh --sample 0.15
```

## Option B — manual step-by-step

Same result as Option A, run one module at a time (useful for debugging a
single stage):

```bash
export PYTHONPATH="src:$PYTHONPATH"
python src/build_dataset.py --full   # M1: raw CSV -> transactions_context.parquet
python src/cleaning.py               # M3: -> transactions_clean.parquet
python src/features.py               # M4: -> models/feature_transformer.joblib
python src/train_validate.py         # M5: -> models/fraud_model.joblib
```

Or via the existing orchestrator (equivalent to `get_data.sh` once the CSV is
already in `data/raw/`):
```bash
./train.sh full      # rebuild everything on 6.36M rows, then train
./train.sh sample 0.15
./train.sh            # train only, reusing the existing processed parquet
```

## Option C — copy pre-built artifacts (skip rebuilding)

If a teammate already ran the pipeline and shared the output files (team
Drive, USB, etc.), just drop them in place and skip straight to later
modules:

| File | Place at | Produced by |
|---|---|---|
| `PS_20174392719_1491204439457_log.csv` | `data/raw/` | Kaggle download |
| `transactions_context.parquet` | `data/processed/` | `src/build_dataset.py` (M1) |
| `transactions_clean.parquet` | `data/processed/` | `src/cleaning.py` (M3) |
| `feature_transformer.joblib` | `models/` | `src/features.py` (M4) |
| `fraud_model.joblib` | `models/` | `src/train_validate.py` (M5) |

Each file lets you skip everything upstream of it. `fraud_model.joblib` alone
is enough to run the API/demo (`uvicorn api.main:app`) without any of the
others — it bundles the model, transformer, threshold, and feature schema.

## Running the app on a fresh clone (no full dataset)

`transactions_context.parquet` is ~526MB on the real full data and is **not**
committed (GitHub's 100MB limit). To keep a bare clone runnable, a small
**stratified sample** `data/processed/context_sample.parquet` (~28MB, ~300k
rows, real fraud rate preserved) **is** committed. The Streamlit analytics
pages (home / evaluation / cost / segment) and the monitoring view fall back to
it automatically when the full frame is absent, and badge themselves as
**"sample mode"** (approximate figures). So right after `git clone` the entire
app runs — the model bundles are committed too.

Rebuild the full frame any time for production-accurate numbers:
```bash
python src/build_dataset.py --full     # -> transactions_context.parquet (526MB)
```
Regenerate the committed sample after rebuilding the full frame:
```bash
python src/make_context_sample.py      # -> context_sample.parquet (~28MB)
```

## Verifying you rebuilt the right thing

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/transactions_clean.parquet')
print(f'rows={len(df):,} fraud={df.isFraud.sum():,} rate={df.isFraud.mean():.4%}')
"
```
Full data expects `rows=6,362,620 fraud=8,213 rate=0.1291%`. A 15% sample
scales proportionally (~954,393 rows, same fraud rate).

## Why these files are gitignored

`transactions_clean.parquet`/`transactions_context.parquet` are ~550-590MB
each — well past GitHub's soft limits for a plain git repo. Regenerating them
locally is fast (a few minutes) and guarantees they're byte-for-byte
consistent with the current code, so it's simpler than trying to version
half-gigabyte binaries. See `.gitignore` for the exact list of what's excluded.
