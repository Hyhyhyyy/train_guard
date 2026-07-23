"""Report writers: HTML and helpers."""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from ..core.io_util import utc_now_iso
from .. import __version__


def html_escape(value: Any) -> str:
    """Escape HTML."""
    return html.escape("" if value is None else str(value), quote=True)


def render_html_report(
    title: str,
    summary_cards: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    disclaimer: str,
) -> str:
    """Single-file self-contained HTML report (no remote assets)."""
    cards_html = []
    for card in summary_cards:
        status = str(card.get("status", "INFO")).upper()
        color = {
            "PASS": "#1b7f4e",
            "WARN": "#b36b00",
            "FAIL": "#b00020",
            "ERROR": "#b00020",
            "INFO": "#1a4b8c",
        }.get(status, "#333")
        cards_html.append(
            f'<div class="card" style="border-left:6px solid {color}">'
            f'<div class="card-title">{html_escape(card.get("title", ""))}</div>'
            f'<div class="card-value">{html_escape(card.get("value", ""))}</div>'
            f'<div class="card-status" style="color:{color}">{html_escape(status)}</div>'
            f"</div>"
        )
    sections_html = []
    for sec in sections:
        headers = sec.get("headers") or ["Item", "Value"]
        thead = "".join(f"<th>{html_escape(h)}</th>" for h in headers)
        rows = []
        for row in sec.get("rows") or []:
            rows.append("<tr>" + "".join(f"<td>{html_escape(c)}</td>" for c in row) + "</tr>")
        note = sec.get("note")
        note_html = f'<p class="note">{html_escape(note)}</p>' if note else ""
        sections_html.append(
            f'<section><h2>{html_escape(sec.get("title", ""))}</h2>{note_html}'
            f'<table><thead><tr>{thead}</tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html_escape(title)}</title>
<style>
body {{ font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; background: #f7f8fa; color: #222; }}
.disclaimer {{ background: #fff3cd; border: 1px solid #ffeeba; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
.card {{ background: #fff; padding: 14px 16px; border-radius: 8px; min-width: 160px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.card-title {{ font-size: 13px; color: #666; }}
.card-value {{ font-size: 22px; font-weight: 600; margin: 6px 0; }}
.card-status {{ font-size: 12px; font-weight: 600; }}
section {{ background: #fff; padding: 16px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; word-break: break-word; }}
th {{ background: #f0f2f5; }}
.note {{ color: #666; font-size: 13px; }}
footer {{ color: #888; font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<h1>{html_escape(title)}</h1>
<p>Generated (UTC): {html_escape(utc_now_iso())}</p>
<div class="disclaimer">{html_escape(disclaimer)}</div>
<div class="cards">{"".join(cards_html)}</div>
{"".join(sections_html)}
<footer>Train Guard v{html_escape(__version__)} — read-only report; paths redacted by default.</footer>
</body>
</html>
"""
