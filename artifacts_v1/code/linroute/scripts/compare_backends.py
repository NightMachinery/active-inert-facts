#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import resource
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

IGNORE_KEYS = frozenset({

    "git_commit",
    "attempt_id",
    "supersedes",
    "timestamp", "timestamps", "ts", "ts_start", "ts_end", "started_at",
    "finished_at", "created", "created_at", "date", "datetime", "wall_clock",

    "elapsed", "elapsed_s", "elapsed_seconds", "seconds", "duration",
    "duration_s", "runtime", "runtime_s", "wall_s", "cpu_s",

    "compute_backend", "compute_device", "compute_dtype",
    "basis_drivers", "basis_driver_counts",

    "host", "hostname", "machine", "probe_host", "probe_machine", "out",
    "rows_out", "run_dir", "arm_dir", "wild_cache_path", "log",
})

IGNORE_PATHS = (
    "basis_fallback_used",
)

MUST_MATCH_LEAF_NAMES = frozenset({
    "available",
    "layers",
    "modal_layer",
    "n",
    "q",
    "n_classes",
    "holdout_rule",
    "protocol",
})

MUST_MATCH_PATHS = (
    "production_eligible",
    "status",
    "core",
)

MUST_MATCH_SUBTREES = (
    "checks",
)

TOLERATED_PATHS: tuple[str, ...] = ()

NOTABLE_ABS = 1e-9
NOTABLE_REL = 1e-9

MAX_THREADS_ON_SHARED_HOST = 4
MAX_CONCURRENT_PROBES = 2
NICENESS = 10

DEFAULT_ARCHIVE = os.environ.get("LINROUTE_ARCHIVE", "archive")
EXPLORATORY_SUBDIR = "analysis-exploratory"

PROBE_FLAGS_WITHOUT_VALUE = frozenset({"--allow-intermediate"})

STRIPPED_FLAGS = ("--out", "--rows-out", "--attempt", "--supersedes",
                  "--threads", "--backend", "--device")

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

@dataclass(frozen=True)
class Leaf:
    path: str
    key: str
    value: Any

def flatten(obj: Any, path: str = "", key: str = "") -> Iterator[Leaf]:
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            child = f"{path}.{k}" if path else str(k)
            yield from flatten(obj[k], child, str(k))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from flatten(item, f"{path}[{i}]", key)
    else:
        yield Leaf(path, key, obj)

def _segments(path: str) -> list[str]:
    out = []
    for part in path.split("."):
        out.append(part.split("[", 1)[0])
    return out

def is_ignored(path: str) -> bool:
    if path in IGNORE_PATHS:
        return True
    for prefix in IGNORE_PATHS:
        if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
            return True
    return any(seg in IGNORE_KEYS for seg in _segments(path))

def is_tolerated(path: str) -> bool:
    for prefix in TOLERATED_PATHS:
        if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
            return True
    return False

def is_must_match(path: str, key: str) -> bool:
    if is_tolerated(path):
        return False
    if key in MUST_MATCH_LEAF_NAMES:
        return True
    segs = _segments(path)
    if segs and segs[0] in MUST_MATCH_SUBTREES:
        return True
    for exact in MUST_MATCH_PATHS:
        if path == exact or path.startswith(exact + ".") or path.startswith(exact + "["):
            return True
    return False

def contract_contradictions() -> list[str]:
    return sorted(IGNORE_KEYS & MUST_MATCH_LEAF_NAMES)

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)

@dataclass
class NumericDelta:
    path: str
    a: float
    b: float
    abs_delta: float
    rel_delta: float
    must_match: bool

@dataclass
class Mismatch:
    path: str
    a: Any
    b: Any
    reason: str

@dataclass
class Comparison:
    label_a: str
    label_b: str
    numeric: list[NumericDelta] = field(default_factory=list)
    equal_non_numeric: int = 0
    equal_numeric: int = 0
    non_numeric_mismatches: list[Mismatch] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    must_match_failures: list[Mismatch] = field(default_factory=list)
    must_match_checked: int = 0
    ignored: int = 0

    @property
    def clean(self) -> bool:
        return not self.must_match_failures

    @property
    def max_abs(self) -> NumericDelta | None:
        finite = [d for d in self.numeric if math.isfinite(d.abs_delta)]
        return max(finite, key=lambda d: d.abs_delta, default=None)

    @property
    def max_rel(self) -> NumericDelta | None:
        finite = [d for d in self.numeric if math.isfinite(d.rel_delta)]
        return max(finite, key=lambda d: d.rel_delta, default=None)

    def to_dict(self) -> dict:
        def d(x: NumericDelta | None) -> dict | None:
            if x is None:
                return None
            return {"path": x.path, "a": x.a, "b": x.b,
                    "abs_delta": x.abs_delta, "rel_delta": x.rel_delta}

        return {
            "a": self.label_a,
            "b": self.label_b,
            "clean": self.clean,
            "must_match_checked": self.must_match_checked,
            "must_match_failures": [vars(m) for m in self.must_match_failures],
            "largest_abs_delta": d(self.max_abs),
            "largest_rel_delta": d(self.max_rel),
            "n_numeric_leaves": len(self.numeric) + self.equal_numeric,
            "n_numeric_differing": len([x for x in self.numeric if x.abs_delta != 0.0]),
            "non_numeric_mismatches": [vars(m) for m in self.non_numeric_mismatches],
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "ignored_leaves": self.ignored,
            "notable_deltas": [
                {"path": x.path, "a": x.a, "b": x.b,
                 "abs_delta": x.abs_delta, "rel_delta": x.rel_delta}
                for x in sorted(self.numeric, key=lambda x: -x.abs_delta)
                if x.abs_delta >= NOTABLE_ABS or x.rel_delta >= NOTABLE_REL
            ][:50],
        }

def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale

def compare_metrics(a: Any, b: Any, label_a: str = "a", label_b: str = "b") -> Comparison:
    cmp = Comparison(label_a=label_a, label_b=label_b)
    leaves_a = {leaf.path: leaf for leaf in flatten(a)}
    leaves_b = {leaf.path: leaf for leaf in flatten(b)}

    for path in sorted(set(leaves_a) | set(leaves_b)):
        if is_ignored(path):
            cmp.ignored += 1
            continue
        la, lb = leaves_a.get(path), leaves_b.get(path)
        key = (la or lb).key
        gated = is_must_match(path, key)
        if gated:
            cmp.must_match_checked += 1

        if la is None or lb is None:
            side = label_b if la is None else label_a
            (cmp.only_in_b if la is None else cmp.only_in_a).append(path)
            if gated:
                cmp.must_match_failures.append(Mismatch(
                    path, None if la is None else la.value,
                    None if lb is None else lb.value,
                    f"present only in {side}"))
            continue

        va, vb = la.value, lb.value
        if _is_number(va) and _is_number(vb):
            abs_d = abs(float(va) - float(vb))
            rel_d = _rel(float(va), float(vb))
            if abs_d == 0.0:
                cmp.equal_numeric += 1
            else:
                cmp.numeric.append(NumericDelta(path, float(va), float(vb),
                                                abs_d, rel_d, gated))
            if gated and abs_d != 0.0:
                cmp.must_match_failures.append(Mismatch(
                    path, va, vb, f"must not differ (abs {abs_d:.6g})"))
        else:
            if va == vb and type(va) is type(vb):
                cmp.equal_non_numeric += 1
            elif va == vb:
                cmp.non_numeric_mismatches.append(Mismatch(
                    path, va, vb, f"equal value, type {type(va).__name__} vs "
                                  f"{type(vb).__name__}"))
                if gated:
                    cmp.must_match_failures.append(Mismatch(
                        path, va, vb, "must not differ (type changed)"))
            else:
                cmp.non_numeric_mismatches.append(Mismatch(path, va, vb, "differs"))
                if gated:
                    cmp.must_match_failures.append(Mismatch(
                        path, va, vb, "must not differ"))
    return cmp

def render_comparison(cmp: Comparison, verbose: bool = False) -> str:
    out: list[str] = []
    out.append(f"A = {cmp.label_a}")
    out.append(f"B = {cmp.label_b}")
    out.append("")
    verdict = "PASS" if cmp.clean else "FAIL"
    out.append(f"verdict: {verdict}   "
               f"({cmp.must_match_checked} gated leaves checked, "
               f"{len(cmp.must_match_failures)} failed)")
    n_num = len(cmp.numeric) + cmp.equal_numeric
    out.append(f"numeric leaves: {n_num} ({len(cmp.numeric)} differ, "
               f"{cmp.equal_numeric} bit-equal)")
    out.append(f"other leaves:   {cmp.equal_non_numeric + len(cmp.non_numeric_mismatches)} "
               f"({len(cmp.non_numeric_mismatches)} differ)")
    out.append(f"ignored leaves: {cmp.ignored}")
    if cmp.only_in_a:
        out.append(f"only in A ({len(cmp.only_in_a)}): "
                   + ", ".join(cmp.only_in_a[:8])
                   + (" ..." if len(cmp.only_in_a) > 8 else ""))
    if cmp.only_in_b:
        out.append(f"only in B ({len(cmp.only_in_b)}): "
                   + ", ".join(cmp.only_in_b[:8])
                   + (" ..." if len(cmp.only_in_b) > 8 else ""))

    out.append("")
    ma, mr = cmp.max_abs, cmp.max_rel
    if ma is None:
        out.append("largest absolute delta: none (no numeric leaf differs)")
    else:
        out.append(f"largest absolute delta: {ma.abs_delta:.6g}  at {ma.path}"
                   f"   (A={ma.a!r}  B={ma.b!r})")
    if mr is None:
        out.append("largest relative delta: none")
    else:
        out.append(f"largest relative delta: {mr.rel_delta:.6g}  at {mr.path}"
                   f"   (A={mr.a!r}  B={mr.b!r})")

    out.append("")
    out.append("must-not-differ gate:")
    if cmp.clean:
        out.append(f"  PASS  all {cmp.must_match_checked} gated leaves identical")
    else:
        for m in cmp.must_match_failures[:60]:
            out.append(f"  FAIL  {m.path}: {m.a!r} -> {m.b!r}  [{m.reason}]")
        if len(cmp.must_match_failures) > 60:
            out.append(f"  ... {len(cmp.must_match_failures) - 60} more")

    notable = [x for x in sorted(cmp.numeric, key=lambda x: -x.abs_delta)
               if x.abs_delta >= NOTABLE_ABS or x.rel_delta >= NOTABLE_REL]
    if notable:
        out.append("")
        out.append(f"largest numeric deltas ({min(len(notable), 20)} of {len(notable)}):")
        for x in notable[:20]:
            out.append(f"  {x.abs_delta:12.6g}  rel {x.rel_delta:12.6g}  {x.path}")
    if verbose and cmp.non_numeric_mismatches:
        out.append("")
        out.append("non-numeric differences:")
        for m in cmp.non_numeric_mismatches[:40]:
            out.append(f"  {m.path}: {m.a!r} -> {m.b!r}")
    bad = contract_contradictions()
    if bad:
        out.append("")
        out.append("WARNING: names both ignored and gated (ignore wins): "
                   + ", ".join(bad))
    return "\n".join(out)

@dataclass(frozen=True)
class Cell:
    host: str
    model: str
    relation: str
    repeat: int

    @property
    def slug(self) -> str:
        return f"{self.host}__{self.model}__{self.relation}__r{self.repeat}"

    def __str__(self) -> str:
        return f"{self.host} {self.model} {self.relation} r{self.repeat}"

def parse_cell(spec: str) -> Cell:
    parts = [p for p in spec.replace("/", " ").replace(",", " ").split() if p]
    if len(parts) != 4:
        raise ValueError(
            f"cell {spec!r} is not 'host model relation repeat' (4 fields, got {len(parts)})")
    host, model, relation, repeat = parts
    return Cell(host, model, relation, int(repeat.lstrip("rR")))

def load_cells(specs: Sequence[str]) -> list[Cell]:
    cells: list[Cell] = []
    for spec in specs:
        candidate = Path(spec.lstrip("@"))
        if spec.startswith("@") or (candidate.exists() and candidate.is_file()):
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    cells.append(parse_cell(line))
        else:
            cells.append(parse_cell(spec))
    return cells

def read_probe_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def _row_matches(row: dict, cell: Cell) -> bool:
    if row.get("relation_slug") != cell.relation:
        return False
    if int(row.get("repeat", -1)) != cell.repeat:
        return False
    if cell.model not in (row.get("model_slug"), row.get("model_name")):
        return False
    if cell.host in ("-", "*", "any"):
        return True
    return cell.host in (row.get("run_host"), row.get("host"))

def find_probe_row(rows: Iterable[dict], cell: Cell) -> dict:
    matches = [(i, r) for i, r in enumerate(rows) if _row_matches(r, cell)]
    if not matches:
        raise LookupError(f"no probes.jsonl row matches {cell}")
    with_cmd = [(i, r) for i, r in matches if r.get("cmd")]
    if not with_cmd:
        raise LookupError(
            f"{len(matches)} probes.jsonl row(s) match {cell} but none carries a cmd")
    with_cmd.sort(key=lambda t: (str(t[1].get("ts_end") or ""),
                                 str(t[1].get("ts_start") or ""),
                                 t[0]))
    return with_cmd[-1][1]

@dataclass
class ParsedCommand:
    cwd: str | None
    env: dict[str, str]
    interpreter: str
    script: str
    args: list[tuple[str, str | None]]

    def flag(self, name: str) -> str | None:
        for f, v in self.args:
            if f == name:
                return v
        return None

def parse_probe_cmd(cmd: str) -> ParsedCommand:
    tokens = shlex.split(cmd)
    if not tokens:
        raise ValueError("empty cmd")
    cwd = None
    if tokens[0] == "cd":
        if len(tokens) < 3 or tokens[2] != "&&":
            raise ValueError(f"cannot parse the 'cd ... &&' prefix of: {cmd[:120]}")
        cwd = tokens[1]
        tokens = tokens[3:]
    while tokens:
        if tokens[0] == "nice":
            tokens = tokens[3:] if len(tokens) > 2 and tokens[1] == "-n" else tokens[1:]
        elif tokens[0] == "env":
            tokens = tokens[1:]
        else:
            break
    env: dict[str, str] = {}
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        name, _, value = tokens[0].partition("=")
        if not name.replace("_", "").isalnum():
            break
        env[name] = value
        tokens = tokens[1:]
    if len(tokens) < 2:
        raise ValueError(f"cmd has no interpreter and script: {cmd[:120]}")
    interpreter, script, rest = tokens[0], tokens[1], tokens[2:]
    if not script.endswith(".py"):
        raise ValueError(f"expected a .py script, got {script!r}")

    args: list[tuple[str, str | None]] = []
    i = 0
    while i < len(rest):
        token = rest[i]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional argument {token!r} in cmd")
        if "=" in token:
            flag, _, value = token.partition("=")
            args.append((flag, value))
            i += 1
            continue
        if token in PROBE_FLAGS_WITHOUT_VALUE:
            args.append((token, None))
            i += 1
            continue
        if i + 1 >= len(rest) or rest[i + 1].startswith("--"):
            args.append((token, None))
            i += 1
            continue
        args.append((token, rest[i + 1]))
        i += 2
    return ParsedCommand(cwd, env, interpreter, script, args)

@dataclass
class ReprobePlan:
    cell: Cell
    backend: str
    device: str
    threads: int
    argv: list[str]
    env: dict[str, str]
    cwd: str | None
    out: Path
    rows_out: Path
    stdout_log: Path
    stderr_log: Path
    original_metrics: Path | None
    source_row_attempt: str | None

    def shell(self) -> str:
        parts = []
        if self.cwd:
            parts.append(f"cd {shlex.quote(self.cwd)} &&")
        parts += [f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items())]
        parts += [shlex.quote(t) for t in self.argv]
        return " ".join(parts)

    def output_paths(self) -> list[Path]:
        return [self.out, self.rows_out, self.stdout_log, self.stderr_log]

def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

def plan_reprobe(cell: Cell, row: dict, *, out_dir: Path, backend: str, device: str,
                 threads: int, repo_dir: str | None = None,
                 python: str | None = None) -> ReprobePlan:
    parsed = parse_probe_cmd(row["cmd"])

    cwd = repo_dir or parsed.cwd
    interpreter = python or parsed.interpreter
    script = parsed.script
    if repo_dir:
        script = str(Path(repo_dir) / "scripts" / Path(parsed.script).name)

    device_slug = device.replace(":", "").replace("/", "")
    leaf = out_dir / f"{backend}-{device_slug}" / cell.host / cell.model / \
        cell.relation / f"r{cell.repeat}"
    out = leaf / "metrics.json"
    rows_out = leaf / "rows.jsonl"

    kept = [(f, v) for f, v in parsed.args if f not in STRIPPED_FLAGS]
    kept.append(("--out", str(out)))
    kept.append(("--rows-out", str(rows_out)))
    kept.append(("--threads", str(threads)))
    kept.append(("--backend", backend))
    kept.append(("--device", device))

    argv = ["nice", "-n", str(NICENESS), interpreter, script]
    for flag, value in kept:
        argv.append(flag)
        if value is not None:
            argv.append(value)

    env = dict(parsed.env)
    for var in THREAD_VARS:
        env[var] = str(threads)

    original = Path(row["out"]) if row.get("out") else None

    plan = ReprobePlan(
        cell=cell, backend=backend, device=device, threads=threads,
        argv=argv, env=env, cwd=cwd, out=out, rows_out=rows_out,
        stdout_log=leaf / "stdout.log", stderr_log=leaf / "stderr.log",
        original_metrics=original,
        source_row_attempt=row.get("attempt_id"),
    )
    _assert_contained(plan, out_dir)
    return plan

def _assert_contained(plan: ReprobePlan, out_dir: Path) -> None:
    for path in plan.output_paths():
        if not _inside(path, out_dir):
            raise ValueError(f"output path {path} is outside --out-dir {out_dir}")
    if plan.original_metrics and plan.out.resolve() == plan.original_metrics.resolve():
        raise ValueError("the rewrite would overwrite the original metrics.json")
    for flag, value in zip(plan.argv, plan.argv[1:]):
        if flag in ("--out", "--rows-out") and not _inside(Path(value), out_dir):
            raise ValueError(f"{flag} {value} is outside --out-dir {out_dir}")
        if flag in ("--attempt", "--supersedes"):
            raise ValueError(f"{flag} must never be reused by this harness")

def assert_writable_out_dir(out_dir: Path, archive: Path) -> None:
    if not _inside(out_dir, archive):
        return
    allowed = archive / EXPLORATORY_SUBDIR
    if not _inside(out_dir, allowed):
        raise SystemExit(
            f"refusing --out-dir {out_dir}: inside the append-only archive but not "
            f"under {allowed}")

@dataclass
class Timing:
    wall_s: float
    cpu_user_s: float
    cpu_sys_s: float
    max_rss_kb: int
    exit_code: int

    @property
    def cpu_s(self) -> float:
        return self.cpu_user_s + self.cpu_sys_s

    def to_dict(self) -> dict:
        return {"wall_s": self.wall_s, "cpu_user_s": self.cpu_user_s,
                "cpu_sys_s": self.cpu_sys_s, "cpu_s": self.cpu_s,
                "max_rss_kb": self.max_rss_kb, "exit_code": self.exit_code}

def run_plan(plan: ReprobePlan) -> Timing:
    plan.out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(plan.env)
    started = time.monotonic()
    with open(plan.stdout_log, "wb") as so, open(plan.stderr_log, "wb") as se:
        proc = subprocess.Popen(plan.argv, cwd=plan.cwd, env=env,
                                stdout=so, stderr=se, stdin=subprocess.DEVNULL)
        _, status, usage = os.wait4(proc.pid, 0)
    wall = time.monotonic() - started

    exit_code = os.waitstatus_to_exitcode(status)
    proc.returncode = exit_code
    return Timing(wall_s=wall, cpu_user_s=usage.ru_utime, cpu_sys_s=usage.ru_stime,
                  max_rss_kb=usage.ru_maxrss, exit_code=exit_code)

def render_timing_table(records: list[dict]) -> str:
    if not records:
        return "no timings recorded"
    lines = ["timing (wall / cpu seconds, cpu:wall ratio):"]
    width = max(len(r["cell"]) for r in records)
    for r in records:
        lines.append(
            f"  {r['cell']:<{width}}  {r['backend']:>6s} {r['device']:<8s} "
            f"wall {r['wall_s']:9.1f}  cpu {r['cpu_s']:9.1f}  "
            f"x{(r['cpu_s'] / r['wall_s'] if r['wall_s'] else 0):5.2f}  "
            f"rss {r['max_rss_kb'] / 1024:8.0f} MiB  exit {r['exit_code']}")
    by_backend: dict[str, list[dict]] = {}
    for r in records:
        by_backend.setdefault(f"{r['backend']}/{r['device']}", []).append(r)
    lines.append("")
    lines.append("summary per backend:")
    for name, rs in sorted(by_backend.items()):
        wall = sum(r["wall_s"] for r in rs)
        cpu = sum(r["cpu_s"] for r in rs)
        lines.append(f"  {name:<16s} {len(rs):2d} cell(s)  total wall {wall:9.1f}s  "
                     f"total cpu {cpu:9.1f}s")
    baselines = [r for r in records if r["backend"] == "numpy"]
    others = [r for r in records if r["backend"] != "numpy"]
    if baselines and others:
        keyed = {(r["cell"]): r for r in baselines}
        lines.append("")
        lines.append("speedup versus the numpy run of the same cell:")
        for r in others:
            base = keyed.get(r["cell"])
            if base and r["wall_s"]:
                lines.append(f"  {r['cell']:<{width}}  wall x{base['wall_s'] / r['wall_s']:.2f}"
                             f"   cpu x{(base['cpu_s'] / r['cpu_s']) if r['cpu_s'] else 0:.2f}")
    return "\n".join(lines)

def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def cmd_compare(a: argparse.Namespace) -> int:
    cmp = compare_metrics(_load_json(a.a), _load_json(a.b),
                          label_a=str(a.a), label_b=str(a.b))
    print(render_comparison(cmp, verbose=a.verbose))
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(cmp.to_dict(), indent=1) + "\n",
                                    encoding="utf-8")
    return 0 if cmp.clean else 1

def cmd_run(a: argparse.Namespace) -> int:
    out_dir = Path(a.out_dir).expanduser()
    archive = Path(a.archive).expanduser()
    assert_writable_out_dir(out_dir, archive)

    threads = min(a.threads, MAX_THREADS_ON_SHARED_HOST)
    if threads != a.threads:
        print(f"note: --threads {a.threads} clamped to {threads} "
              f"(shared host rule)", file=sys.stderr)
    parallel = min(a.max_parallel, MAX_CONCURRENT_PROBES)

    probes = Path(a.probes_jsonl) if a.probes_jsonl else archive / "status" / "probes.jsonl"
    if not probes.exists():
        print(f"cannot read {probes}", file=sys.stderr)
        return 2
    rows = read_probe_rows(probes)
    cells = load_cells(a.cells)
    if not cells:
        print("no cells given", file=sys.stderr)
        return 2

    backends = [a.backend]
    if a.also_baseline and a.backend != "numpy":
        backends = ["numpy", a.backend]

    plans: list[ReprobePlan] = []
    for cell in cells:
        try:
            row = find_probe_row(rows, cell)
        except LookupError as exc:
            print(f"SKIP {cell}: {exc}", file=sys.stderr)
            continue
        for backend in backends:
            device = "cpu" if backend == "numpy" else a.device
            plans.append(plan_reprobe(
                cell, row, out_dir=out_dir, backend=backend, device=device,
                threads=threads, repo_dir=a.repo_dir, python=a.python))

    if not plans:
        print("nothing to do", file=sys.stderr)
        return 2

    print(f"# {len(plans)} re-probe(s), {parallel} at a time, "
          f"{threads} thread(s) each, nice {NICENESS}")
    for plan in plans:
        print()
        print(f"# cell {plan.cell}  backend {plan.backend} device {plan.device}")
        print(f"# original: {plan.original_metrics}")
        print(f"# new out : {plan.out}")
        print(plan.shell())
    if a.dry_run:
        print()
        print("# --dry-run: nothing was executed, nothing was written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    timings: list[dict] = []
    results: list[tuple[ReprobePlan, Timing]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        for plan, timing in zip(plans, pool.map(run_plan, plans)):
            results.append((plan, timing))

    failed = False
    for plan, timing in results:
        record = {"cell": str(plan.cell), "backend": plan.backend,
                  "device": plan.device, "threads": plan.threads,
                  **timing.to_dict()}
        timings.append(record)
        (plan.out.parent / "timing.json").write_text(
            json.dumps(record, indent=1) + "\n", encoding="utf-8")
        if timing.exit_code != 0:
            failed = True
            print(f"\nFAILED {plan.cell} [{plan.backend}] exit {timing.exit_code}; "
                  f"see {plan.stderr_log}", file=sys.stderr)
            continue
        if not plan.original_metrics or not plan.original_metrics.exists():
            print(f"\n{plan.cell} [{plan.backend}]: no original metrics to compare "
                  f"({plan.original_metrics})", file=sys.stderr)
            continue
        cmp = compare_metrics(_load_json(plan.original_metrics), _load_json(plan.out),
                              label_a=str(plan.original_metrics), label_b=str(plan.out))
        text = render_comparison(cmp, verbose=a.verbose)
        print()
        print(f"=== {plan.cell}  [{plan.backend}/{plan.device}] ===")
        print(text)
        (plan.out.parent / "compare.txt").write_text(text + "\n", encoding="utf-8")
        (plan.out.parent / "compare.json").write_text(
            json.dumps(cmp.to_dict(), indent=1) + "\n", encoding="utf-8")
        if not cmp.clean:
            failed = True

    print()
    print(render_timing_table(timings))
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=1) + "\n",
                                          encoding="utf-8")
    return 1 if failed else 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_backends.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode")

    c = sub.add_parser("compare", help="compare two metrics.json files")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json-out", default=None)
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("run", help="re-probe cells on a backend and compare")
    r.add_argument("--cells", action="append", default=[], metavar="SPEC",
                   help="'host model relation repeat', or a file of such lines "
                        "(repeatable)")
    r.add_argument("--backend", default="torch", choices=("numpy", "torch"))
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--threads", type=int, default=4,
                   help=f"BLAS threads per probe (clamped to "
                        f"{MAX_THREADS_ON_SHARED_HOST} on the shared host)")
    r.add_argument("--max-parallel", type=int, default=1,
                   help=f"concurrent probes (capped at {MAX_CONCURRENT_PROBES})")
    r.add_argument("--archive", default=DEFAULT_ARCHIVE)
    r.add_argument("--probes-jsonl", default=None,
                   help="override the probes.jsonl to recover commands from")
    r.add_argument("--repo-dir", default=None,
                   help="run probe_transfer.py from THIS checkout instead of the "
                        "frozen one the original command named")
    r.add_argument("--python", default=None, help="interpreter to run it with")
    r.add_argument("--also-baseline", action="store_true",
                   help="also re-probe each cell with --backend numpy on the same "
                        "code, so a code change and a backend change are separable")
    r.add_argument("--dry-run", action="store_true",
                   help="print the rewritten commands and exit")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)
    return p

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run"] + argv
    a = build_parser().parse_args(argv)
    if not getattr(a, "func", None):
        build_parser().print_help()
        return 2
    return a.func(a)

if __name__ == "__main__":
    raise SystemExit(main())
