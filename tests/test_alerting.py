"""Tests for outbound alerting helpers (no network)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerting import build_alert_payload, incident_report_md, send_webhook  # noqa: E402


def test_build_alert_payload():
    p = build_alert_payload(["ip_billing_distance_km", "num_failed_payment_attempts"],
                            {"ip_billing_distance_km": 0.42}, received=1234, when="2026-07-21T10:00:00")
    assert p["text"].startswith("🚨")
    assert "ip_billing_distance_km" in p["text"]
    assert p["signals"] == ["ip_billing_distance_km", "num_failed_payment_attempts"]
    assert p["received"] == 1234 and p["when"] == "2026-07-21T10:00:00"


def test_incident_report_md():
    md = incident_report_md(["amount"], {"amount": 0.3}, received=10, when="t")
    assert "Retrain Recommended" in md
    assert "| amount | 0.300 |" in md


def test_send_webhook_rejects_bad_url():
    ok, msg = send_webhook("", {"text": "x"})
    assert ok is False and "URL" in msg
    ok2, _ = send_webhook("not-a-url", {"text": "x"})
    assert ok2 is False
