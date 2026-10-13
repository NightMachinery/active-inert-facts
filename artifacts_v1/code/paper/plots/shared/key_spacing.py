from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACING = json.loads((HERE / "key-spacing.json").read_text(encoding="utf-8"))
WIDTHS = json.loads((HERE / "font-widths.json").read_text(encoding="utf-8"))
TAG = re.compile(r"<(/?)(g|text)\b([^>]*)>", re.S)
TEXT = re.compile(r"<text\b([^>]*)>(.*?)</text>", re.S)
ATTR = lambda name: re.compile(rf'\b{name}="([^"]*)"')

def _attr(attrs: str, name: str, default=None):
    m = ATTR(name).search(attrs)
    return m.group(1) if m else default

def _width(text: str, size: float, bold: bool) -> float:
    table = WIDTHS["bold" if bold else "regular"]
    return sum(table.get(ch, WIDTHS["fallback_em"]) for ch in text) * size

def _unesc(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")

def _texts(svg: str):
    spans = []
    depth, start, clear = 0, None, False
    for m in TAG.finditer(svg):
        closing, tag, attrs = m.group(1), m.group(2), m.group(3)
        if tag != "g":
            continue
        if not closing:
            if start is None and "data-key-block=" in attrs:
                start, depth, clear = m.start(), 1, 'data-key-clear="1"' in attrs
            elif start is not None:
                depth += 1
        elif start is not None:
            depth -= 1
            if depth == 0:
                spans.append((start, m.end(), clear))
                start = None
    out = []
    for m in TEXT.finditer(svg):
        attrs, body = m.group(1), _unesc(m.group(2))
        if "rotate" in (_attr(attrs, "transform", "") or ""):
            continue
        x, y = float(_attr(attrs, "x", 0)), float(_attr(attrs, "y", 0))
        size = float(_attr(attrs, "font-size", 8))
        w = _width(body, size, _attr(attrs, "font-weight") in ("700", "bold"))
        anchor = _attr(attrs, "text-anchor", "start")
        x0 = x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
        span = next((i for i, (a, b, _) in enumerate(spans) if a <= m.start() < b), None)
        out.append({"x0": x0, "x1": x0 + w, "y": y, "size": size, "span": span,
                    "clear": span is not None and spans[span][2]})
    return out

def count(svg: str) -> int:
    return svg.count('data-key-clear="1"')

def check(svg: str, name: str = "svg") -> list[str]:
    texts = _texts(svg)
    problems = []
    blocks = {}
    for t in texts:
        if t["clear"]:
            blocks.setdefault(t["span"], []).append(t)
    for span, own in blocks.items():
        top = min(t["y"] for t in own)
        first = [t for t in own if abs(t["y"] - top) < 0.01]
        x0, x1 = min(t["x0"] for t in own), max(t["x1"] for t in own)
        size = first[0]["size"]
        above = [t for t in texts if t["span"] is None and t["y"] < top - 0.01 and t["x1"] > x0 and t["x0"] < x1]
        if not above:
            continue
        near = max(above, key=lambda t: t["y"])
        gap = (top - SPACING["cap_em"] * size) - (near["y"] + SPACING["descent_em"] * near["size"])
        if gap < SPACING["clear_pt"] - 0.05:
            problems.append(f"{name}: key block {span} has {gap:.1f} pt clear space under the text at y={near['y']:.1f}; "
                            f"needs {SPACING['clear_pt']} (plots/shared/key-spacing.json)")
    return problems
