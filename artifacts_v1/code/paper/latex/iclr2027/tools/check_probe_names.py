#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.argv = [str(HERE / "gen_probe_names.py"), "--check", *sys.argv[1:]]
    runpy.run_path(str(HERE / "gen_probe_names.py"), run_name="__main__")
