# Design — Multi-model scoring + monitoring dashboard

Date: 2026-07-21
Status: approved for implementation

## Goal

Enhance the fraud-detection application so that:

1. **All three models** (Logistic Regression, Random Forest, XGBoost) score every
   incoming request **independently** and the API returns each model's result
   **plus a combined `max-risk` verdict**.
2. The **Streamlit review queue** shows the three models side by side.
3. A new **monitoring dashboard page** is added to the Streamlit app, with a
   Reports tab and a Live-recompute tab, showing **per-model** prediction drift.

## Decisions (locked)

| Topic | Decision |
|---|---|
| Response shape | Independent per-model results **+** aggregate verdict |
| Aggregate rule | **max-risk** — most severe decision wins (`block > review > allow`) |
| Model storage | One `models/fraud_ensemble.joblib`, shared transformer, per-model matrix/features/threshold |
| Feature group | All three models on **`realistic`** (non-leaky, authorization-time) |
| Endpoint | **Replace** `POST /score` to return the 3-model + aggregate payload |
| Streamlit | Show 3 models in the queue; add a Monitoring page |
| Dashboard data | Two tabs — **Reports** (rendered artifacts) and **Live** (recompute) |
| Prediction drift | **Per-model** (`_logreg / _rf / _xgb`) on the Live tab and in `drift.py` |
| Backward compat | Still produce single `fraud_model.joblib` so `infer.py` keeps working |

## Components

### 1. `src/ensemble.py` (new) — shared scoring, one source of truth
- `SEVERITY = {"allow":0, "review":1, "block":2}`
- `decide(prob, review_thr, block_floor=0.9) -> str` — block if `prob >= max(block_floor, review_thr)`, review if `prob >= review_thr`, else allow.
- `aggregate_maxrisk(decisions: list[str]) -> str` — highest severity; ignores `None`/errored entries; defaults to `"allow"` if the list is empty.
- `load_ensemble(path) -> dict` — load + validate the ensemble bundle.
- `score_all(enriched_df, bundle, use_dest_history) -> dict[str, np.ndarray]` — per model: `transformer.transform(enriched, entry["matrix"])[entry["features"]]` → `predict_proba[:,1]`. Enrichment is done by the caller once (API=record, Streamlit=batch); this function only transforms + predicts per model.

Ensemble bundle shape:
```
{ transformer, rule:"max-risk", trained_at, split_info,
  models: { logreg:{model,model_name,matrix,features,threshold},
            rf:{...}, xgb:{...} } }
```

### 2. `src/train_validate.py` — emit the ensemble bundle
During the existing comparison loop, when `group == "realistic"`, stash each
model's fitted object, matrix, feature columns, and cost-tuned threshold. After
the loop, assemble and `joblib.dump` `fraud_ensemble.joblib` containing every
model trained on `realistic`. The existing `fraud_model.joblib` (single best) is
still written unchanged.

### 3. `api/main.py` — 3-model `/score`
Load the ensemble at startup. Request body unchanged. Flow: `_ensure_required_columns`
→ `enrich(use_dest_history=False)` (single live record) → `score_all` →
`decide` per model → `aggregate_maxrisk`. Per-model failures are caught: that
model's entry becomes `{"error": ...}`, aggregate uses the survivors, and a
`degraded` flag is set. `GET /` lists the loaded models.

Response:
```json
{ "models": { "logreg": {"model_name","fraud_probability","decision","review_threshold","block_threshold"},
              "rf": {...}, "xgb": {...} },
  "aggregate": {"decision":"review","rule":"max-risk"} }
```

### 4. `monitoring/drift.py` — per-model drift, shared function
Extract `compute_drift(ref_df, cur_df, ensemble_bundle) -> DataFrame` returning
feature PSI rows **plus** one `PREDICTION_SCORE_<key>` row per model (scored via
`ensemble.score_all`). CLI `main()` uses it (writes the `.md` + Evidently html),
now with per-model rows. `psi`, `band`, `inject_drift` stay reusable.

### 5. Streamlit — multipage (`st.navigation`)
- `app/streamlit_app.py` — entrypoint: cached ensemble loader + `st.navigation`
  wiring for two pages.
- `app/review_view.py` — Review Queue: score the batch with all 3 models
  (dest-history enabled), add per-model score columns + `agg_decision` (max-risk);
  metrics/sort use the aggregate.
- `app/monitoring_view.py` — Monitoring page with two tabs:
  - **Reports**: render `drift_report.md`, embed `evidently_drift.html`; friendly
    notice + timestamp when absent.
  - **Live**: reference = `day_index ≤ median`, current = later half; a
    "Simulate fraud campaign" toggle applies `inject_drift`; shows the PSI table
    (features + per-model prediction rows) color-banded, and a retrain banner.

## Build order
1. `ensemble.py` (+ tests) → 2. `train_validate.py` → retrain → 3. `api/main.py`
→ 4. `drift.py` → 5. Streamlit pages → 6. end-to-end verify.

## Testing
- `decide` boundaries (allow/review/block, block floor).
- `aggregate_maxrisk` across combinations incl. degraded (errored/`None`) entries and empty.
- Ensemble bundle has an entry per trained realistic model, each with required keys + valid threshold in [0,1].
- API integration (`TestClient`): benign → aggregate `allow`; fraud-like defaults → `review`/`block`; payload has all 3 model blocks.
- `compute_drift`: one prediction row per model; injected-campaign pushes ≥1 row to `SIGNIFICANT` while the natural split stays `stable`.
- Parity: `score_all` gives identical decisions for a record whether scored singly (API path) or in a batch (Streamlit path).

## Out of scope (YAGNI)
Per-request model selection, weighted/learned ensembles, auto-triggered retraining,
auth, and any change to M1–M4 data pipeline.
