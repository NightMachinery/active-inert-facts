#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine_generated import MG, PAPER, header

COUNTS = PAPER / "notation" / "counts.tex"

READ: list[Path] = []

PRODUCTION = ("data/label-pools-extra/"
              "animal-leaves-sense1hyponym-profanity-zipf232-jointfirsttoken-"
              "jointoffset")
PARENT = ("data/label-pools-extra/"
          "animal-leaves-sense1hyponym-profanity-zipf232-jointfirsttoken")
V2 = "data/label-pools-v2"
FLOOR = 86
NUMWORDS = {8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}

def load(path: Path) -> dict:
    READ.append(path)
    return json.loads(path.read_text(encoding="utf-8"))

def counts_macro(name: str) -> int:
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(\d+)\}", COUNTS.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"notation/counts.tex defines no literal \\{name}")
    return int(m.group(1))

def pool_size(directory: Path, prov: dict) -> int:
    READ.append(directory / "pool.txt")
    lines = [l for l in (directory / "pool.txt").read_text().splitlines()
             if l.strip()]
    if len(lines) != prov["n_labels"]:
        raise SystemExit(f"{directory}: pool.txt has {len(lines)} labels, "
                         f"provenance says {prov['n_labels']}; refusing")
    return len(lines)

def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"

def fmt_count(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")

def chain_table(root: Path) -> str:
    d = root / PRODUCTION
    p = load(d / "provenance.json")
    par_dir = root / PARENT
    par = load(par_dir / "provenance.json")
    if p["parent"]["pool_txt_sha256"] != par["pool_txt_sha256"]:
        raise SystemExit(f"{d}: parent sha256 disagrees with {par_dir}")
    lad = par["ladder"]
    ff = par["frequency_floor"]
    prof = par["rule_2_external_profanity_list"]["list"]
    nd = par["near_duplicate_filter"]
    jf = par["joint_first_token"]
    n = pool_size(d, p)
    assert lad["final_after_cosine"] == pool_size(par_dir, par)
    n_offset_tok = len(p["offset_rule"]["evidence"]["per_model"])
    if n_offset_tok != counts_macro("nFleetModels"):
        raise SystemExit(f"offset_rule measures {n_offset_tok} models, notation/counts.tex says "
                         f"\\nFleetModels {counts_macro('nFleetModels')}; recheck the offset row")
    rows = [
        (r"Enumerate: every lemma string of \texttt{noun.animal} matching "
         r"\texttt{\textasciicircum[a-z]+\$}, deduplicated",
         r"WordNet~3.0", lad["step3_lowercase_alpha"]),
        (r"Leaves: drop a label that is a WordNet noun "
         r"ancestor of another label in the label set", "--", lad["leaves"]),
        (r"Primary WordNet sense is " + tt("animal.n.01") +
         r" or a hyponym of it",
         r"WordNet~3.0 sense order", lad["primary_sense_under_animal"]),
        (r"Not named by a published profanity list",
         r"LDNOOBW English, commit \texttt{" + prof["commit"][:7] + "}",
         lad["after_profanity"]),
        (r"Frequency floor: \texttt{wordfreq} zipf $\geq " +
         f"{ff['threshold']:.2f}$",
         r"\texttt{wordfreq}~" + ff["resource"]["version"] +
         r" (\texttt{large})",
         lad["after_frequency_floor"]),
        (r"First token distinct from every kept label's, under all " +
         NUMWORDS[len(jf["per_tokenizer_alone"])] + r" tokenizers at once",
         r"tokenizer revisions of record", lad["after_joint_first_token"]),
        (r"Near-duplicate: first-token \texttt{lm\_head} "
         r"cosine to a kept label at most " + f"{nd['threshold_cosine']:.3f}",
         r"\MiniCpmModel{}",
         lad["final_after_cosine"]),
        (r"Offset: in every \newrelAdj{} prompt, the label's first token is "
         r"the token the trainer reads, under every roster tokenizer",
         r"checked when each run is built", n),
    ]
    body = "\n".join(
        f"    {i} & {rule} & {pin} & {fmt_count(k)} \\\\"
        for i, (rule, pin, k) in enumerate(rows))
    if n != counts_macro("productionPoolSize"):
        raise SystemExit(f"animal label set has {n} labels, notation/counts.tex says "
                         f"\\productionPoolSize {counts_macro('productionPoolSize')}; refusing")
    return rf"""% Rows of tab:label-pool-chain (results/tables/label-pool-chain.tex).
% Source: linroute {PARENT}/provenance.json
%   (ladder, frequency_floor, rule_2_external_profanity_list,
%    joint_first_token, near_duplicate_filter) for rows 0-6, and
%   {PRODUCTION}/provenance.json (offset_rule, n_labels) for row 7;
%   animal label set pool.txt sha256 {p['pool_txt_sha256']}.
\newcommand{{\labelPoolChainRows}}{{%
{body}
}}
"""

def pools_table(root: Path) -> str:
    rows = []
    prod_dir = root / PRODUCTION
    pp = load(prod_dir / "provenance.json")
    prod_n = pool_size(prod_dir, pp)
    prod_z = pp["measured_not_selected_on"]["zipf_frequency"]["median"]
    prod_nn = pp["measured_not_selected_on"]["panel"]["embedding_raw"][
        "draw_mean_nearest_neighbour"]

    for d in sorted((root / V2).glob("*-singletoken")):
        if d.name.startswith("animal-"):
            continue
        p = load(d / "provenance.json")
        n = pool_size(d, p)
        if p["eligible_ge_86"] != (n >= FLOOR):
            raise SystemExit(f"{d}: eligible_ge_86 disagrees with its size")
        m = p["measured_not_selected_on"]
        z = m["zipf_frequency"]["median"] if n else None
        nn = (m["panel"]["embedding_raw"]["draw_mean_nearest_neighbour"]
              if n and "panel" in m else None)
        rows.append((p["category_word"], p["root_synset"]["synset"], n, z, nn))
    tops_missing = not any(r[0] == "Tops" for r in rows)

    def cell(v, f):
        return "--" if v is None else f(v)

    rows.sort(key=lambda r: r[0])
    lines = []
    for word, rootsyn, n, z, nn in rows:
        label = rf"\texttt{{{word}}}"
        lines.append(
            f"    {label} & {tt(rootsyn)} & {n} & "
            f"{'yes' if n >= FLOOR else 'no'} & "
            f"{cell(z, lambda v: f'{v:.2f}')} & "
            f"{cell(nn, lambda v: f'{v:.3f}'.lstrip('0'))} \\\\")
    eligible = sum(1 for r in rows if r[2] >= FLOOR)
    body = "\n".join(lines)
    note_tops = ("% noun.Tops has no row: its category word resolves to senses"
                 " of 'top', none filed in noun.Tops, so it has no root.\n"
                 if tops_missing else "")
    if FLOOR != counts_macro("poolEligibilityFloor"):
        raise SystemExit(f"FLOOR {FLOOR} differs from notation/counts.tex \\poolEligibilityFloor")
    return rf"""% Rows of tab:label-pools (results/tables/label-pools.tex).
% Sources: linroute {PRODUCTION}/provenance.json (animal row);
%   {V2}/<category>-leaves-sense1hyponym-profanity-zipf232-singletoken/
%   provenance.json (category rows: root_synset, n_labels checked against
%   pool.txt, eligible_ge_86 checked against the size,
%   measured_not_selected_on).
{note_tops}% {eligible} category label sets clear the floor of {FLOOR}.
\newcommand{{\labelPoolsRows}}{{%
    \texttt{{animal}} & {tt('animal.n.01')} & {prod_n} & yes & {prod_z:.2f} & {f'{prod_nn:.3f}'.lstrip('0')} \\
    \midrule
{body}
}}
"""

def render(root: Path, name: str, build) -> str:
    READ.clear()
    body = build(root)
    digest = hashlib.sha256()
    for path in READ:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    src = f"linroute data/label-pools-extra, data/label-pools-v2 ({len(READ)} provenance and pool files, combined)"
    return header("latex/iclr2027/tools/gen_label_pool_tables.py", [(src, digest.hexdigest())],
                  "cd latex/iclr2027 && python3 tools/gen_label_pool_tables.py --linroute <linroute checkout>") + body

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--linroute", type=Path, default=Path(__file__).resolve().parents[4] / "linroute")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = args.linroute.expanduser()
    outputs = {MG / "label-pool-chain-rows.tex": render(root, "chain", chain_table),
               MG / "label-pools-rows.tex": render(root, "pools", pools_table)}
    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text() != text]
    if args.check:
        if stale:
            raise SystemExit(f"gen_label_pool_tables: stale {', '.join(p.name for p in stale)}; rerun without --check")
        print("gen_label_pool_tables: OK, both rows files match the provenance")
        return
    for path, text in outputs.items():
        path.write_text(text)
        print(f"wrote {path}")

if __name__ == "__main__":
    main()
