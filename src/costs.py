"""Business cost model — single source of truth for the €-impact of decisions.

Fraud detection is a cost-optimization problem, not an accuracy problem. This
module turns model decisions + ground-truth labels + transaction amounts into a
money breakdown, using the SAME cost assumptions as the CLI evaluator
(`src/infer.py::print_eval_metrics`) and the report, all sourced from
`config.py`. Keeping one implementation means the app, the CLI, and the written
report can never disagree on what the model is worth.

Convention
----------
A transaction is *flagged* when its decision is `review` or `block` (positive
prediction). The cost components mirror `infer.print_eval_metrics`:

  * missed fraud (FN)  -> COST_FALSE_NEGATIVE x amount lost
  * false alarm  (FP)  -> COST_FALSE_POSITIVE x count(legit & flagged)
  * review labour      -> COST_MANUAL_REVIEW x count(flagged)

Blocked/reviewed fraud is treated as *loss prevented* (the transaction is
stopped or caught in manual review before the money leaves).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, COST_MANUAL_REVIEW

# Re-export the config defaults so callers have one import site.
DEFAULT_FN = float(COST_FALSE_NEGATIVE)   # weight per unit of amount lost to missed fraud
DEFAULT_FP = float(COST_FALSE_POSITIVE)   # flat currency cost of one false alarm
DEFAULT_REVIEW = float(COST_MANUAL_REVIEW)  # flat currency cost of one manual review


def decisions_from_risk(risk: np.ndarray, review_thr: float, block_thr: float) -> np.ndarray:
    """Map an ensemble risk score to allow / review / block.

    block if risk >= block_thr; review if risk >= review_thr; else allow. The
    block gate is clamped to never sit below the review gate.
    """
    risk = np.asarray(risk, dtype=float)
    block_thr = max(block_thr, review_thr)
    out = np.full(len(risk), "allow", dtype=object)
    out[risk >= review_thr] = "review"
    out[risk >= block_thr] = "block"
    return out


@dataclass
class CostResult:
    """Money breakdown for one operating point. All amounts in currency units."""
    n: int
    n_fraud: int
    n_flagged: int
    # confusion (positive = flagged)
    tp: int          # fraud, flagged
    fp: int          # legit, flagged
    fn: int          # fraud, allowed  (missed)
    tn: int          # legit, allowed
    # money
    fraud_exposure: float     # total € of fraud in the population
    caught_amount: float      # € of fraud prevented (flagged frauds)
    missed_amount: float      # € of fraud lost (allowed frauds)
    fn_loss: float            # COST_FALSE_NEGATIVE x missed_amount
    fp_cost: float            # COST_FALSE_POSITIVE x fp
    review_cost: float        # COST_MANUAL_REVIEW x n_flagged
    model_cost: float         # fn_loss + fp_cost + review_cost
    do_nothing_cost: float    # allow everything -> lose all fraud
    review_all_cost: float    # review every txn -> only labour, catch all fraud
    net_savings: float        # do_nothing_cost - model_cost
    savings_vs_review_all: float  # review_all_cost - model_cost

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def loss_avoided_pct(self) -> float:
        return self.caught_amount / self.fraud_exposure * 100 if self.fraud_exposure else 0.0

    @property
    def roi(self) -> float:
        """Return per unit of operating spend (FP + review cost)."""
        spend = self.fp_cost + self.review_cost
        return self.net_savings / spend if spend else float("inf")


def cost_breakdown(
    y_true: np.ndarray,
    flagged: np.ndarray,
    amount: np.ndarray,
    *,
    c_fn: float = DEFAULT_FN,
    c_fp: float = DEFAULT_FP,
    c_review: float = DEFAULT_REVIEW,
) -> CostResult:
    """Compute the full money breakdown for a flagged/allowed decision vector.

    y_true  : 0/1 ground-truth fraud labels
    flagged : bool array, True where the txn was flagged (review or block)
    amount  : transaction amounts (same length)
    """
    y = np.asarray(y_true).astype(int)
    f = np.asarray(flagged).astype(bool)
    a = np.asarray(amount, dtype=float)

    is_fraud = y == 1
    tp = int(np.sum(is_fraud & f))
    fp = int(np.sum(~is_fraud & f))
    fn = int(np.sum(is_fraud & ~f))
    tn = int(np.sum(~is_fraud & ~f))

    fraud_exposure = float(a[is_fraud].sum())
    caught_amount = float(a[is_fraud & f].sum())
    missed_amount = float(a[is_fraud & ~f].sum())

    fn_loss = c_fn * missed_amount
    fp_cost = c_fp * fp
    review_cost = c_review * int(f.sum())
    model_cost = fn_loss + fp_cost + review_cost

    do_nothing_cost = c_fn * fraud_exposure           # flag nothing: lose all fraud
    review_all_cost = c_review * len(y)               # review everyone: labour only

    return CostResult(
        n=len(y), n_fraud=int(is_fraud.sum()), n_flagged=int(f.sum()),
        tp=tp, fp=fp, fn=fn, tn=tn,
        fraud_exposure=fraud_exposure, caught_amount=caught_amount, missed_amount=missed_amount,
        fn_loss=fn_loss, fp_cost=fp_cost, review_cost=review_cost, model_cost=model_cost,
        do_nothing_cost=do_nothing_cost, review_all_cost=review_all_cost,
        net_savings=do_nothing_cost - model_cost,
        savings_vs_review_all=review_all_cost - model_cost,
    )


def sweep_threshold(
    y_true: np.ndarray,
    risk: np.ndarray,
    amount: np.ndarray,
    *,
    c_fn: float = DEFAULT_FN,
    c_fp: float = DEFAULT_FP,
    c_review: float = DEFAULT_REVIEW,
    steps: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Net savings as a function of the flag threshold on the risk score.

    Returns (thresholds, net_savings) so the caller can plot the curve and mark
    the cost-optimal operating point.
    """
    thresholds = np.linspace(0.0, 1.0, steps)
    savings = np.array([
        cost_breakdown(y_true, np.asarray(risk) >= t, amount,
                       c_fn=c_fn, c_fp=c_fp, c_review=c_review).net_savings
        for t in thresholds
    ])
    return thresholds, savings


def optimal_threshold(
    y_true: np.ndarray,
    risk: np.ndarray,
    amount: np.ndarray,
    **kwargs,
) -> tuple[float, float]:
    """(threshold, net_savings) at the cost-optimal flag threshold."""
    thresholds, savings = sweep_threshold(y_true, risk, amount, **kwargs)
    i = int(np.argmax(savings))
    return float(thresholds[i]), float(savings[i])
