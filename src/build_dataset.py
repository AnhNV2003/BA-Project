"""Orchestrator: PaySim base data -> + synthetic context -> processed dataset.

Usage
-----
    python src/build_dataset.py            # default: 15% stratified sample
    python src/build_dataset.py --frac 0.3 # custom stratified fraction
    python src/build_dataset.py --full     # full 6.36M rows

Outputs
-------
    data/processed/transactions_context.parquet
"""
from __future__ import annotations

import argparse
import time

from config import DATA_PROCESSED
from data_base import load_base_data
from synth_context import FIELD_META, add_synthetic_context


def build(frac: float | None = 0.15, full: bool = False) -> None:
    started = time.perf_counter()
    frac = None if full else frac

    base = load_base_data(sample_frac=frac)
    base_cols = set(base.columns)
    aug = add_synthetic_context(base)

    out_parquet = DATA_PROCESSED / "transactions_context.parquet"
    try:
        aug.to_parquet(out_parquet, index=False)
        saved = out_parquet
    except Exception as exc:
        saved = out_parquet.with_suffix(".csv")
        aug.to_csv(saved, index=False)
        print(f"[build] parquet unavailable ({exc}); wrote CSV")

    synth_cols = [c for c in aug.columns if c not in base_cols]
    elapsed = time.perf_counter() - started
    print("\n=== PAYSIM BUILD SUMMARY ===")
    print(f"rows        : {len(aug):,}")
    print(f"columns     : {aug.shape[1]}  ({len(synth_cols)} synthetic/context added)")
    print(f"fraud       : {int(aug['isFraud'].sum()):,}  ({aug['isFraud'].mean():.4%})")
    print(f"synth cols  : {', '.join([c for c in FIELD_META if c in aug.columns])}")
    print(f"saved       : {saved.relative_to(saved.parents[2])}")
    print(f"elapsed     : {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=0.15, help="stratified sample fraction")
    parser.add_argument("--full", action="store_true", help="use the entire PaySim dataset")
    args = parser.parse_args()
    build(frac=args.frac, full=args.full)
