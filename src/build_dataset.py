"""Orchestrator: base PaySim -> + synthetic context -> processed dataset.

Usage
-----
    python build_dataset.py            # default: ~15% stratified sample (draft/model)
    python build_dataset.py --frac 0.3 # custom stratified fraction
    python build_dataset.py --full     # entire 6.36M rows (heavier; ~2-3 min)

Outputs (data/processed/)
    transactions_context.parquet   full augmented dataset used for modelling
    sample_preview.csv             2,000-row human-readable preview (for demo/review)

The sample is STRATIFIED on isFraud so the real ~0.13% prevalence is preserved
(honest evaluation), while keeping enough positives to train on.
"""
from __future__ import annotations

import argparse
import time

from config import DATA_PROCESSED
from data_base import load_base_data
from synth_context import add_synthetic_context


def build(frac: float | None = 0.15, full: bool = False) -> None:
    t0 = time.perf_counter()
    frac = None if full else frac

    base = load_base_data(sample_frac=frac)
    aug = add_synthetic_context(base)

    out_parquet = DATA_PROCESSED / "transactions_context.parquet"
    try:
        aug.to_parquet(out_parquet, index=False)
        saved = out_parquet
    except Exception as e:                       # parquet engine missing
        saved = out_parquet.with_suffix(".csv")
        aug.to_csv(saved, index=False)
        print(f"[build] parquet unavailable ({e}); wrote CSV")

    preview = DATA_PROCESSED / "sample_preview.csv"
    aug.sample(n=min(2000, len(aug)), random_state=0).to_csv(preview, index=False)

    dt = time.perf_counter() - t0
    print("\n=== BUILD SUMMARY ===")
    print(f"rows        : {len(aug):,}")
    print(f"columns     : {aug.shape[1]}  ({aug.shape[1] - 11} synthetic added)")
    print(f"fraud       : {int(aug['isFraud'].sum()):,}  ({aug['isFraud'].mean():.4%})")
    print(f"saved       : {saved.relative_to(saved.parents[2])}")
    print(f"preview     : {preview.relative_to(preview.parents[2])}")
    print(f"elapsed     : {dt:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.15, help="stratified sample fraction")
    ap.add_argument("--full", action="store_true", help="use the entire dataset")
    args = ap.parse_args()
    build(frac=args.frac, full=args.full)
