"""Tests for the API Tester HTTP client + presets (no live server needed)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from api_client import health, score  # noqa: E402


def test_score_rejects_bad_url():
    ok, msg = score("", {"type": "PAYMENT"})
    assert ok is False and "URL" in msg
    ok2, _ = score("ftp://x", {"type": "PAYMENT"})
    assert ok2 is False


def test_health_rejects_bad_url():
    ok, msg = health("nope")
    assert ok is False


def test_score_unreachable_returns_error_not_raise():
    # nothing listening on this port → returns (False, msg), never raises
    ok, msg = score("http://127.0.0.1:9", {"type": "PAYMENT"}, timeout=1.0)
    assert ok is False and isinstance(msg, str)


def test_presets_cover_all_fields():
    import api_tester_view as v
    for name, preset in v.PRESETS.items():
        keys = set(preset)
        assert keys == set(v._FIELDS), f"preset {name!r} field mismatch: {keys ^ set(v._FIELDS)}"
        assert preset["type"] in v._TYPES
