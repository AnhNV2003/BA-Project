"""Unit tests for the shared ensemble scoring logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensemble import aggregate_maxrisk, block_threshold, decide  # noqa: E402


@pytest.mark.parametrize(
    "prob, review, expected",
    [
        (0.00, 0.10, "allow"),
        (0.09, 0.10, "allow"),
        (0.10, 0.10, "review"),   # boundary: at review threshold -> review
        (0.50, 0.10, "review"),
        (0.89, 0.10, "review"),   # just under block floor -> review
        (0.90, 0.10, "block"),    # boundary: at block floor -> block
        (0.99, 0.10, "block"),
    ],
)
def test_decide_boundaries(prob, review, expected):
    assert decide(prob, review) == expected


def test_decide_block_floor_never_below_review():
    # A high review threshold raises the block gate with it.
    assert block_threshold(0.95) == 0.95
    assert decide(0.92, 0.95) == "allow"    # below review threshold (0.95)
    assert decide(0.95, 0.95) == "block"    # at review==block gate -> block
    assert decide(0.96, 0.95) == "block"


def test_decide_low_review_uses_default_block_floor():
    assert block_threshold(0.10) == 0.9
    assert decide(0.5, 0.10) == "review"
    assert decide(0.9, 0.10) == "block"


@pytest.mark.parametrize(
    "decisions, expected",
    [
        (["allow", "allow", "allow"], "allow"),
        (["allow", "review", "allow"], "review"),
        (["allow", "allow", "block"], "block"),
        (["review", "block", "review"], "block"),
        (["review", "review", "review"], "review"),
    ],
)
def test_aggregate_maxrisk(decisions, expected):
    assert aggregate_maxrisk(decisions) == expected


def test_aggregate_ignores_errored_and_empty():
    # None / unknown entries (a model that errored) are ignored.
    assert aggregate_maxrisk(["allow", None, "review"]) == "review"
    assert aggregate_maxrisk([None, "block"]) == "block"
    assert aggregate_maxrisk([]) == "allow"
    assert aggregate_maxrisk([None, None]) == "allow"


def test_decide_089_is_review_not_block():
    # sanity: just under the 0.9 block floor stays review when review_thr is low
    assert decide(0.89, 0.10) == "review"
