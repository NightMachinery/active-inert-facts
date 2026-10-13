#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import arms

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tokenizer", action="append", required=True,
                   metavar="HF_ID",
                   help="a model or tokenizer to measure; repeatable")
    p.add_argument("--animals", default="data/animals/curated.txt", type=Path,
                   help="the declared pool, in file order (default: the "
                   "curated 78)")
    p.add_argument("--target-format", default=arms.TARGET_FORMAT,
                   help="how a label is rendered as a target; must end in "
                   f"'{{animal}}' (default: {arms.TARGET_FORMAT!r})")
    p.add_argument("--json", type=Path, default=None,
                   help="also write the whole pool block, per tokenizer, here")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="one 'model<TAB>|P_M|' line per tokenizer and nothing "
                   "else, for a shell to read")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="also name every dropped name and what absorbed it")
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    animals = arms.load_animals(args.animals)
    blocks = {}
    for hf_id in args.tokenizer:
        block = arms.label_pool(hf_id, animals,
                               target_format=args.target_format)
        blocks[block["tokenizer"]] = block
        if args.quiet:
            print(f"{block['tokenizer']}\t{block['n_usable']}")
            continue
        print(f"{block['tokenizer']}: |P_M| = {block['n_usable']} of "
              f"{block['n_declared']} declared "
              f"({block['n_single_token']} single-token, "
              f"{block['n_multi_token']} multi-token)")
        if block["dropped"]:
            print(f"  pass --max-classes {block['n_usable']} to "
                  f"scripts/make_split_manifest.py for this model")
        if args.verbose:
            for d in block["dropped"]:
                print(f"    dropped {d['animal']!r}: first token "
                      f"{d['first_token']!r} (id {d['first_token_id']}) is "
                      f"already {d['absorbed_by']!r}'s")
    if args.json is not None:
        args.json.write_text(json.dumps(blocks, indent=1, ensure_ascii=False)
                             + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
