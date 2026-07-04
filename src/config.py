"""Central configuration: paths, seed, schema, and business-cost assumptions.

Keeping these in one place means every module (data gen, EDA, modeling,
serving) shares the same definitions — important for reproducibility and for
the data dictionary / report to stay in sync with the code.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the repo root, so the code runs from anywhere)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SYNTH = ROOT / "data" / "synthetic"
DATA_PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
MODELS = ROOT / "models"

for _p in (DATA_RAW, DATA_SYNTH, DATA_PROCESSED, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# Base (PaySim) schema — the columns the real Kaggle CSV must contain.
# We use this to (a) validate the real file when it arrives, and
# (b) generate a schema-identical stand-in so the pipeline runs before the
# download is done.
# ----------------------------------------------------------------------------
PAYSIM_COLUMNS = [
    "step",            # unit of time = 1 hour of simulation (1..743 ~ 30 days)
    "type",            # PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT
    "amount",          # transaction amount
    "nameOrig",        # customer initiating the transaction (C...)
    "oldbalanceOrg",   # origin balance before the transaction
    "newbalanceOrig",  # origin balance after the transaction
    "nameDest",        # recipient (C... customer, M... merchant)
    "oldbalanceDest",  # destination balance before
    "newbalanceDest",  # destination balance after
    "isFraud",         # ground-truth fraud label (0/1)  <-- TARGET
    "isFlaggedFraud",  # PaySim's own naive rule flag (transfer > 200k)
]
TARGET = "isFraud"

# In real PaySim, fraud only ever occurs in these two transaction types.
# Our stand-in replicates this so EDA shows the real structural pattern.
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# ----------------------------------------------------------------------------
# Business-cost assumptions  (Module 5 — cost-based threshold selection)
# These are ASSUMPTIONS the team must justify in the report; tune them here so
# threshold selection and the report never drift apart.
# ----------------------------------------------------------------------------
# Cost of a missed fraud (false negative): we lose (roughly) the transaction
# amount. Modeled as a multiple of amount at scoring time, but for a simple
# fixed-cost view we assume an average ticket loss.
COST_FALSE_NEGATIVE = 1.0     # weight per unit of transaction amount lost
# Cost of a false alarm (false positive): customer friction, manual-review
# labor, possible churn. A flat cost per wrongly-blocked legit transaction.
COST_FALSE_POSITIVE = 25.0    # currency units per false positive
# Cost of manually reviewing a flagged transaction (review-queue capacity).
COST_MANUAL_REVIEW = 3.0

# Fraud prevalence we target in the stand-in (real PaySim ~= 0.0013).
STANDIN_FRAUD_RATE = 0.0015
STANDIN_N_ROWS = 100_000
