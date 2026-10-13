#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

SUPERLATIVES = [
    "narrowest", "widest", "largest", "smallest", "highest", "lowest",
    "strongest", "weakest", "best", "worst", "most", "least", "greatest",
    "tightest", "loosest", "biggest", "maximum", "minimum", "dominant",
    "outperforms", "outperform", "beats", "first",
]

QUANTIFIERS = [
    "only", "every", "all", "none", "no model", "no relation", "no cell",
    "never", "always", "each", "both", "neither", "either", "entire",
    "whole", "exactly", "unique", "uniquely", "sole", "solely",
    "universally", "without exception", "in no ", "not one", "any model",
    "any relation", "nothing", "everything", "invariably", "throughout",
]

EST_STOP = {
    "test", "tests", "tested", "interest", "interests", "latest", "modest",
    "honest", "request", "requests", "suggest", "suggests", "suggested",
    "rest", "protest", "manifest", "harvest", "nearest", "closest",
    "earliest", "west", "best",
}

SUPPRESSORS = [
    ("least", re.compile(r"\bat\s+least\b", re.I)),
    ("most", re.compile(r"\bat\s+most\b", re.I)),
    ("most", re.compile(r"\bmost\s+(?:of\s+)?(?:the\s+)?(?:likely|"
                        r"important|recent)\b", re.I)),
    ("first", re.compile(r"\bfirst[- ]order\b", re.I)),
    ("first", re.compile(r"\bfirst\s+letter\b", re.I)),
]

NOUNS = (r"models?|rows?|cells?|relations?|refits?|draws?|checks?|"
         r"coefficients?|columns?|points?|intervals?|fits?|entries|entry|"
         r"groups?|repeats?|subjects?|seeds?|labels?|pools?|classes|class|"
         r"templates?|paraphrases?|tokenizers?|panels?|folds?")

_UNITS = r"one|two|three|four|five|six|seven|eight|nine"
_TEENS = (r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
          r"eighteen|nineteen")
_TENS = r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"

NUMWORD = (rf"(?:{_TENS})[-\s](?:{_UNITS})|(?:{_TENS})|(?:{_TEENS})|"
           rf"(?:two|three|four|five|six|seven|eight|nine)|"
           rf"hundred|thousand")

COUNTMACRO = r"\\n[A-Z][A-Za-z]*(?:\{\})?"

SCOPE_PATTERNS = [
    re.compile(rf"{COUNTMACRO}\s+(?:{NOUNS})\b"),
    re.compile(rf"\b(?:of|among|amongst|over|across|within|for|in)\s+the\s+"
               rf"{COUNTMACRO}"),
    re.compile(r"\bn\s*=\s*\d+"),
    re.compile(r"\bdf\s*=\s*[-+]?\d+"),
    re.compile(rf"\b\d+\s+(?:{NOUNS})\b", re.I),
    re.compile(rf"\b(?:{NUMWORD})\s+(?:{NOUNS})\b", re.I),
    re.compile(rf"\b(?:of|among|amongst|over|across|within|for|in)\s+the\s+"
               rf"(?:{NUMWORD}|\d+)\b", re.I),
    re.compile(r"\bof\s+(?:the\s+)?\d+\b", re.I),
    re.compile(r"\bper\s+model\b", re.I),
    re.compile(r"\b(?:in|for)\s+(?:this|that)\s+"
               r"(?:row|column|entry|group|cohort|model|relation)\b", re.I),
    re.compile(r"\b(?:restricted|limited|confined)\s+to\b", re.I),
    re.compile(r"\bat\s+n\s*=", re.I),
]

COUNTED = (r"(?!\s+(?:\d+|(?:" + NUMWORD + r"))\b)"
           r"(?!\s+(?:the\s+)?" + COUNTMACRO + r")")

UNBOUNDED = [
    r"anywhere", r"in\s+the\s+table",
    r"of\s+any\b" + COUNTED,
    r"of\s+all\b" + COUNTED,
    r"\bever\b", r"in\s+the\s+paper",

    r"in\s+this\s+file", r"\boverall\b", r"across\s+the\s+board",
    r"in\s+every\s+table", r"any\s+of\s+the" + COUNTED, r"\bno\s+other\b",
    r"than\s+any\b", r"in\s+existence",
]
UNBOUNDED_RE = [(u, re.compile(u, re.I)) for u in UNBOUNDED]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+")

def build_term_regex() -> re.Pattern:
    parts = [re.escape(t) for t in SUPERLATIVES + QUANTIFIERS]
    parts.append(r"[A-Za-z]{3,}est\b")
    return re.compile(r"(?<![A-Za-z])(" + "|".join(parts) + r")(?![A-Za-z])",
                      re.I)

TERMS = build_term_regex()

def strip_tex_comment(line: str) -> str:
    out = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)

IGNORE_LINE = re.compile(r"scope-checker:\s*ignore\b(?!-)", re.I)
IGNORE_BEGIN = re.compile(r"scope-checker:\s*ignore-begin\b", re.I)
IGNORE_END = re.compile(r"scope-checker:\s*ignore-end\b", re.I)

MARKER_TAIL = re.compile(r"\s*(?:[#%]+\s*)?scope-checker:\s*ignore\b(?!-).*$",
                         re.I)

def ignored_lines(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ignored, markers, open_at = set(), [], None
    for number, line in enumerate(raw, start=1):
        if IGNORE_BEGIN.search(line):
            if open_at is not None:
                markers.append((number, "ERROR: ignore-begin already open at "
                                f"line {open_at}"))
            open_at = number
            markers.append((number, "ignore-begin"))
            continue
        if IGNORE_END.search(line):
            if open_at is None:
                markers.append((number, "ERROR: ignore-end with no begin"))
            else:
                ignored.update(range(open_at, number + 1))
                open_at = None
                markers.append((number, "ignore-end"))
            continue
        if IGNORE_LINE.search(line):
            ignored.add(number)
            markers.append((number, "ignore (one line)"))
    if open_at is not None:
        markers.append((open_at, "ERROR: ignore-begin never closed; it would "
                                 "have silenced the rest of the file"))
        ignored.update(range(open_at, len(raw) + 1))
    return ignored, markers

def read_units(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = []
    for number, line in enumerate(raw, start=1):
        text = strip_tex_comment(line) if path.suffix == ".tex" else line

        if IGNORE_BEGIN.search(text) or IGNORE_END.search(text):
            text = ""
        else:
            text = MARKER_TAIL.sub("", text)
        lines.append((number, text))

    frags = []

    def joined_text():
        return " ".join(f[2] for f in frags)

    def span_of(lo, hi):
        touched = [f[0] for f in frags
                   if f[1] < hi and f[1] + len(f[2]) > lo]
        if not touched:
            touched = [frags[0][0]] if frags else [1]
        return min(touched), max(touched)

    def push(number, text):
        offset = 0 if not frags else frags[-1][1] + len(frags[-1][2]) + 1
        frags.append((number, offset, text))

    for number, text in lines:
        stripped = text.strip()
        if not stripped:
            if frags:
                joined = joined_text()
                lo, hi = span_of(0, len(joined))
                yield lo, hi, joined
                frags.clear()
            continue
        push(number, stripped)
        joined = joined_text()
        if not SENTENCE_SPLIT.search(joined + " "):
            continue
        cursor = 0
        pieces = []
        for match in SENTENCE_SPLIT.finditer(joined):
            pieces.append((cursor, match.start()))
            cursor = match.end()
        for lo_off, hi_off in pieces:
            piece = joined[lo_off:hi_off]
            if piece.strip():
                lo, hi = span_of(lo_off, hi_off)
                yield lo, hi, piece
        if cursor:
            kept = []
            for line_number, offset, frag_text in frags:
                frag_end = offset + len(frag_text)
                if frag_end <= cursor:
                    continue
                cut = max(0, cursor - offset)
                remainder = frag_text[cut:]
                if remainder.strip() or kept:
                    kept.append((line_number, remainder))
            frags.clear()
            for line_number, remainder in kept:
                push(line_number, remainder)
            while frags and not frags[0][2].strip():
                dropped = frags.pop(0)
                rebuilt = [(n, t) for n, _, t in frags]
                frags.clear()
                for n, t in rebuilt:
                    push(n, t)
    if frags:
        joined = joined_text()
        lo, hi = span_of(0, len(joined))
        yield lo, hi, joined

def classify(sentence: str):
    hits = sorted({m.group(0).lower() for m in TERMS.finditer(sentence)
                   if m.group(0).lower() not in EST_STOP})
    for term, pattern in SUPPRESSORS:
        if term in hits and pattern.search(sentence):
            hits.remove(term)
    if not hits:
        return None
    unbounded = [u for u, pattern in UNBOUNDED_RE if pattern.search(sentence)]
    scoped = any(p.search(sentence) for p in SCOPE_PATTERNS)
    if unbounded:
        verdict = "UNSCOPED-HIGH"
    elif scoped:
        verdict = "SCOPED"
    else:
        verdict = "UNSCOPED"
    return verdict, hits, unbounded

def default_targets(root: Path):
    targets = []
    registry = root / "sections" / "results-numbers.org"
    if registry.exists():
        targets.append(registry)
    targets.extend(sorted(root.rglob("*.tex")))
    return targets

def main(argv):
    root = Path(__file__).resolve().parent.parent
    targets = [Path(a) for a in argv[1:]] or default_targets(root)
    if not targets:
        print("no files to scan; is the tree laid out as expected?")
        return 0

    counts = {"SCOPED": 0, "UNSCOPED": 0, "UNSCOPED-HIGH": 0}
    scanned = 0
    per_file = {}
    suppressed = []
    marker_errors = []

    for path in targets:
        if not path.exists():
            print(f"missing: {path}")
            continue
        scanned += 1
        exempt, markers = ignored_lines(path)
        marker_errors.extend((path, number, what) for number, what in markers
                             if what.startswith("ERROR"))
        for number, end, sentence in read_units(path):
            verdict = classify(sentence)
            if verdict is None:
                continue
            kind, hits, unbounded = verdict
            if kind != "SCOPED" and exempt & set(range(number, end + 1)):
                excerpt = re.sub(r"\s+", " ", sentence).strip()
                suppressed.append((path, number, kind, hits,
                                   excerpt[:160]))
                continue
            counts[kind] += 1
            if kind == "SCOPED":
                continue
            excerpt = re.sub(r"\s+", " ", sentence).strip()
            if len(excerpt) > 240:
                excerpt = excerpt[:237] + "..."
            per_file.setdefault(path, []).append(
                (number, kind, hits, unbounded, excerpt))

    for path in sorted(per_file, key=str):
        print(f"\n=== {path}")
        for number, kind, hits, unbounded, excerpt in per_file[path]:
            flag = f" unbounded={unbounded}" if unbounded else ""
            print(f"{path}:{number}: {kind} terms={hits}{flag}")
            print(f"    {excerpt}")

    if suppressed:
        print("\n=== suppressed by an explicit scope-checker marker")
        print("    These are NOT clean; somebody exempted them.  Read the")
        print("    list every run: this is the only thing standing between")
        print("    an exemption and a habit.")
        for path, number, kind, hits, excerpt in suppressed:
            print(f"{path}:{number}: SUPPRESSED {kind} terms={hits}")
            print(f"    {excerpt}")

    total = sum(counts.values())
    print()
    print("-" * 70)
    print(f"files scanned                 {scanned}")
    print(f"claims matched                {total}")
    print(f"  carrying a scope clause     {counts['SCOPED']}")
    print(f"  NO scope clause             {counts['UNSCOPED']}")
    print(f"  NO scope, unbounded phrase  {counts['UNSCOPED-HIGH']}")
    print(f"suppressed by a marker        {len(suppressed)}")
    print()
    if marker_errors:
        print("MARKER ERRORS.  A malformed marker can silence a whole file,")
        print("which is the failure this tool exists to catch, so this exits")
        print("non-zero even though findings themselves are advisory.")
        for path, number, what in marker_errors:
            print(f"{path}:{number}: {what}")
        print()
        return 2
    if total == 0:
        print("Zero matches. That is more likely a parsing failure than a")
        print("clean paper; check that the files were found and read.")
    else:
        print("This is a prompt to re-read, not a verdict. Re-run whenever the")
        print("population changes: a clean run at one population says nothing")
        print("about the next one.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
