from __future__ import annotations

import re

TAG = re.compile(r"<(rect|line|path)\b[^>]*>", re.S)
LAYER = re.compile(r'\bdata-layer="(band|grid)"')
ROW_FILLS = ("#F7F7F8", "#f7f7f7", "#F3F4F6")

def check(svg: str, name: str = "svg") -> list[str]:
    problems: list[str] = []
    seen_grid = False
    for m in TAG.finditer(svg):
        el = m.group(0)
        layer = LAYER.search(el)
        if layer is None:
            if m.group(1) == "rect" and any(f'fill="{c}"' in el for c in ROW_FILLS) and 'height="' in el:
                w = re.search(r'width="([\d.]+)"', el)
                h = re.search(r'height="([\d.]+)"', el)
                if w and h and float(w.group(1)) > 4 * float(h.group(1)):
                    problems.append(f"{name}: untagged row stripe {el[:90]}")
            continue
        if layer.group(1) == "grid":
            seen_grid = True
        elif seen_grid:
            problems.append(f"{name}: a row band is painted after a gridline, so it covers it: {el[:90]}")
    return problems
