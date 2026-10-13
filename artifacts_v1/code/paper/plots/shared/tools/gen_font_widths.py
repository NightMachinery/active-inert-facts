#!/usr/bin/env python3
import json
import subprocess
import sys

from fontTools.ttLib import TTFont

RANGES = [(0x20, 0x7E), (0xA0, 0x17F), (0x370, 0x3FF), (0x2000, 0x206F), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x25A0, 0x25FF)]

def widths(style):
    path = subprocess.run(["fc-match", "-f", "%{file}", f"Liberation Sans:{style}"], capture_output=True, text=True, check=True).stdout
    font = TTFont(path)
    upm = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]
    out = {}
    for lo, hi in RANGES:
        for cp in range(lo, hi + 1):
            if cp in cmap:
                out[chr(cp)] = round(hmtx[cmap[cp]][0] / upm, 4)
    return path.rsplit("/", 1)[-1], out

reg_file, reg = widths("regular")
bold_file, bold = widths("bold")
json.dump({"note": "Advance widths in em of Liberation Sans (Regular, Bold), from plots/shared/tools/gen_font_widths.py; characters not covered fall back to fallback_em.",
           "fonts": [reg_file, bold_file], "fallback_em": 0.6, "regular": reg, "bold": bold}, sys.stdout, ensure_ascii=False, indent=0, sort_keys=True)
sys.stdout.write("\n")
