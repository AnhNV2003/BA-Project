"""Shared setup for the Streamlit app: path wiring + cached resource loaders."""
from __future__ import annotations

import pathlib
import sys

import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "monitoring"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from ensemble import load_ensemble          # noqa: E402
from config import MODELS, DATA_PROCESSED   # noqa: E402
import drift                                 # noqa: E402

REPORTS = ROOT / "monitoring" / "reports"
ENSEMBLE_PATH = MODELS / "fraud_ensemble.joblib"


@st.cache_resource
def get_ensemble() -> dict:
    return load_ensemble(ENSEMBLE_PATH)


@st.cache_data
def get_context_data():
    """Full context frame (for the monitoring temporal split)."""
    return drift.load()


def model_keys(bundle: dict) -> list[str]:
    return list(bundle["models"].keys())
