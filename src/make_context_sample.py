"""Build a small, committable fallback for the demo app.

`data/processed/transactions_context.parquet` is ~526MB on the real full PaySim
data — too big for git. The Streamlit analytics pages (home/eval/cost/segment)
and the monitoring view read that frame. This script writes a much smaller
`context_sample.parquet` (~300k rows, ~26MB) that is committed to the repo so a
fresh clone can run the whole app without rebuilding the full pipeline.

The sample is STRATIFIED on `isFraud`, so it preserves the true fraud rate
(~0.1291%) — the descriptive Segment page reports honest per-segment rates, and
the Eval/Cost pages get a real (if smaller, approximate) test split. Pages that
fall back to this file badge themselves as running in "sample mode".

Regenerate:  PYTHONPATH=src python src/make_context_sample.py [target_rows]
"""
from __future__ import annotations

import sys

import pandas as pd

from config import DATA_PROCESSED, SEED

TARGET_ROWS = 300_000
SRC = DATA_PROCESSED / "transactions_context.parquet"
OUT = DATA_PROCESSED / "context_sample.parquet"


def build(target_rows: int = TARGET_ROWS) -> None:
    if not SRC.exists():
        raise FileNotFoundError(
            f"{SRC} not found — build the full dataset first "
            "(python src/build_dataset.py --full)."
        )
    df = pd.read_parquet(SRC)
    frac = min(1.0, target_rows / len(df))

    # Stratify on the label so the sample keeps the real fraud prevalence.
    sample = (
        df.groupby("isFraud", group_keys=False)
        .sample(frac=frac, random_state=SEED)
        .reset_index(drop=True)
    )
    sample.to_parquet(OUT, index=False)

    size_mb = OUT.stat().st_size / 1e6
    print("=== CONTEXT SAMPLE ===")
    print(f"source rows : {len(df):,}")
    print(f"sample rows : {len(sample):,}  (frac={frac:.4f})")
    print(f"fraud       : {int(sample['isFraud'].sum()):,}  ({sample['isFraud'].mean():.4%})")
    print(f"saved       : {OUT.relative_to(OUT.parents[2])}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_ROWS
    build(rows)
