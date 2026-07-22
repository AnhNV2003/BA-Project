"""Outbound drift/retrain alerts — generic incoming webhook + incident report.

Nothing is sent unless the caller supplies a URL. The payload uses a top-level
`text` field so it works with Slack and Discord incoming webhooks as well as any
generic JSON endpoint. Uses only the stdlib (urllib) — no extra dependency.
"""
from __future__ import annotations

import json
import urllib.request


def build_alert_payload(signals, psi: dict | None = None, received=None, when=None) -> dict:
    """Slack/Discord-compatible payload describing the retrain condition."""
    psi = psi or {}
    bullets = "\n".join(
        f"• {s}: PSI {psi[s]:.3f}" if isinstance(psi.get(s), (int, float)) else f"• {s}"
        for s in signals
    )
    text = ("🚨 *Fraud model — retrain recommended*\n"
            f"{len(list(signals))} signal(s) exceeded the drift threshold:\n{bullets}")
    if received is not None:
        text += f"\n(after {received:,} transactions)"
    return {
        "text": text,
        "signals": list(signals),
        "psi": {s: psi.get(s) for s in signals},
        "received": received,
        "when": when,
    }


def incident_report_md(signals, psi: dict | None = None, received=None, when=None) -> str:
    """A downloadable Markdown incident report."""
    psi = psi or {}
    lines = [
        "# Drift Incident — Retrain Recommended",
        "",
        f"- When: {when or 'n/a'}",
        f"- Transactions processed: {received:,}" if received is not None else "- Transactions processed: n/a",
        f"- Signals over threshold: {len(list(signals))}",
        "",
        "| Signal | PSI |",
        "|---|---|",
    ]
    for s in signals:
        v = psi.get(s)
        lines.append(f"| {s} | {v:.3f} |" if isinstance(v, (int, float)) else f"| {s} | - |")
    return "\n".join(lines) + "\n"


def send_webhook(url: str, payload: dict, timeout: float = 5.0) -> tuple[bool, str]:
    """POST `payload` as JSON to `url`. Returns (ok, message). Never raises."""
    if not url or not str(url).lower().startswith(("http://", "https://")):
        return False, "No valid http(s) webhook URL configured."
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"Delivered (HTTP {resp.status})."
    except Exception as exc:  # network/DNS/HTTP errors — surface, never crash the app
        return False, f"{type(exc).__name__}: {exc}"
