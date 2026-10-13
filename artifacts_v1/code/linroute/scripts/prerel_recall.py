#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

LOGITS_SCHEMA = "linroute/prerel-cand-logits/1"
READOUT_SCHEMA = "linroute/prerel-recall/1"
CAPTURED_PASS = "rel_subject"

def _repo_from_argv(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--":
            break
        if arg == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if arg.startswith("--repo="):
            return Path(arg.split("=", 1)[1]).resolve()
    return Path(os.environ.get("PRE_REL_REPO") or Path(__file__).resolve().parents[1])

def use_repo(repo: Path) -> None:
    for p in (str(repo / "scripts"), str(repo)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    loaded = sys.modules.get("linroute")
    if loaded is not None:
        where = Path(loaded.__file__).resolve().parents[1]
        if where != repo:
            raise SystemExit(f"linroute already imported from {where}, not {repo}")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

class Tap:
    def __init__(self) -> None:
        self.passes: list[tuple] = []

    def install(self) -> None:
        import torch
        from linroute import models

        inner_span = models.span_states
        inner_forward = models.forward_pass
        tap = self

        def span_states(model, tokenizer, encoded, cand_ids, **kw):
            if kw.get("name") != CAPTURED_PASS:
                return inner_span(model, tokenizer, encoded, cand_ids, **kw)
            chunks: list = []
            cand_dev: dict = {}

            def forward_pass(m, input_ids, attention_mask):
                hidden, logits = inner_forward(m, input_ids, attention_mask)
                dev = logits.device
                if dev not in cand_dev:
                    cand_dev[dev] = cand_ids.to(dev)
                chunks.append(
                    logits.index_select(1, cand_dev[dev]).float().cpu().clone()
                )
                return hidden, logits

            models.forward_pass = forward_pass
            try:
                states = inner_span(model, tokenizer, encoded, cand_ids, **kw)
            finally:
                models.forward_pass = inner_forward
            logits = (torch.cat(chunks, 0) if chunks
                      else torch.empty(0, len(cand_ids), dtype=torch.float32))
            tap.check(states, logits)
            tap.passes.append((cand_ids.clone(), logits))
            return states

        models.span_states = span_states

    @staticmethod
    def check(states, logits) -> None:
        import torch

        if tuple(logits.shape) != tuple(states.cand_logits.shape):
            raise SystemExit(
                f"capture: {tuple(logits.shape)} logits for a pass whose "
                f"cand_logits is {tuple(states.cand_logits.shape)}"
            )
        if not torch.equal(logits.argmax(-1), states.cand_top1):
            bad = int((logits.argmax(-1) != states.cand_top1).sum())

            ties = int(((logits == logits.max(-1, keepdim=True).values)
                        .sum(-1) > 1).sum())
            if bad > ties:
                raise SystemExit(
                    f"capture: argmax of the captured logits differs from the "
                    f"pass's cand_top1 on {bad} prompts ({ties} exact ties)"
                )
        if not torch.equal(logits.to(torch.float16), states.cand_logits):
            raise SystemExit(
                "capture: the captured logits do not round to the pass's own "
                "fp16 cand_logits; they are not the same tensor"
            )

def capture(argv: list[str]) -> int:
    if "--" not in argv:
        raise SystemExit("capture: pass the builder's arguments after --")
    cut = argv.index("--")
    ours, builder_argv = argv[:cut], argv[cut + 1:]
    p = argparse.ArgumentParser(prog="prerel_recall.py capture")
    p.add_argument("--repo", required=True)
    p.add_argument("--builder", required=True,
                   help="repo-relative builder script, e.g. scripts/cache_wild_pack.py")
    p.add_argument("--logits-out", required=True)
    a = p.parse_args(ours)
    repo = Path(a.repo).resolve()
    builder = (repo / a.builder).resolve()
    if builder.parent != repo / "scripts" or not builder.is_file():
        raise SystemExit(f"capture: {builder} is not a script of {repo}")
    out = Path(a.logits_out).resolve()
    if out.exists():
        raise SystemExit(f"capture: {out} exists; this step is create-only")

    import numpy as np
    import cache_wild

    tap = Tap()
    tap.install()
    written: list[dict] = []
    inner_relation = cache_wild.cache_relation

    def cache_relation(model, tokenizer, rel, **kw):
        before = len(tap.passes)
        data = inner_relation(model, tokenizer, rel, **kw)
        new = tap.passes[before:]
        if len(new) != 1:
            raise SystemExit(f"capture: {len(new)} {CAPTURED_PASS} passes for {rel.name!r}")
        cand_ids, logits = new[0]
        n, t = len(rel.subjects), len(rel.templates)
        logits = logits.view(n, t, -1)
        zs = data["zs_cand"]
        if not (logits.argmax(-1) == zs).all() and not _ties_explain(logits, zs):
            raise SystemExit("capture: captured argmax disagrees with zs_cand")
        if not np.array_equal(np.asarray(cand_ids), np.asarray(data["cand_first_token"])):
            raise SystemExit("capture: candidate ids differ from cand_first_token")
        written.append({
            "relation": rel.name,
            "subjects": list(rel.subjects),
            "templates": list(rel.templates),
            "cand_logits": logits.numpy().astype(np.float32),
            "zs_cand": np.asarray(zs, dtype=np.int64),
            "cand_first_token": np.asarray(cand_ids, dtype=np.int64),
            "cache_created": data["meta"]["created"],
            "model": data["meta"]["model"],
        })
        return data

    cache_wild.cache_relation = cache_relation

    sys.argv = [str(builder)] + builder_argv
    importlib.import_module(builder.stem).main(builder_argv)
    if len(written) != 1:
        raise SystemExit(
            f"capture: the builder cached {len(written)} relations; this step "
            f"expects exactly one (an existing cache is skipped by the builder "
            f"and leaves nothing to capture)"
        )
    w = written[0]
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        schema=np.array(LOGITS_SCHEMA),
        relation=np.array(w["relation"]),
        model=np.array(w["model"]),
        cache_created=np.array(w["cache_created"]),
        subjects=np.array(w["subjects"], dtype=object).astype(str),
        templates=np.array(w["templates"], dtype=object).astype(str),
        cand_logits=w["cand_logits"],
        zs_cand=w["zs_cand"],
        cand_first_token=w["cand_first_token"],
    )
    os.replace(tmp, out)
    print(f"CAPTURE_OK {w['relation']!r} logits={tuple(w['cand_logits'].shape)} -> {out}",
          flush=True)
    return 0

def _ties_explain(logits, zs) -> bool:
    top = logits.max(-1, keepdim=True).values
    bad = logits.argmax(-1) != zs
    at_max = logits.gather(-1, zs.unsqueeze(-1)).squeeze(-1) == top.squeeze(-1)
    return bool(at_max[bad].all())

def recall_readout(cache: dict, logits: dict, manifest: dict) -> tuple[dict, dict]:
    import numpy as np
    from linroute.probes import _class_macro_mean
    from linroute.transfer import CANONICAL_TEMPLATE

    subjects = list(cache["subjects"])
    if list(logits["subjects"]) != subjects:
        raise ValueError("captured logits and cache disagree on the subject list")
    if list(logits["templates"]) != list(cache["rel_templates"]):
        raise ValueError("captured logits and cache disagree on the templates")
    zs = np.asarray(cache["zs_cand"], dtype=np.int64)
    if not np.array_equal(np.asarray(logits["zs_cand"]), zs):
        raise ValueError("captured zs_cand differs from the cache's: not the same build")
    cand_tok = np.asarray(cache["cand_first_token"], dtype=np.int64)
    if not np.array_equal(np.asarray(logits["cand_first_token"]), cand_tok):
        raise ValueError("captured candidate ids differ from the cache's")
    L = np.asarray(logits["cand_logits"], dtype=np.float32)
    n, t = zs.shape
    if L.shape != (n, t, len(cand_tok)):
        raise ValueError(f"logits shape {L.shape} != {(n, t, len(cand_tok))}")
    arg = L.argmax(-1)
    top = L.max(-1)
    tie_ok = np.take_along_axis(L, zs[..., None], -1)[..., 0] == top
    argmax_mismatch = int(((arg != zs) & ~tie_ok).sum())
    if argmax_mismatch:
        raise ValueError(f"{argmax_mismatch} zs_cand entries are not a maximum of the logits")

    pos = {s: i for i, s in enumerate(subjects)}
    split = np.array(["-"] * n, dtype="<U1")
    subject_id = np.array([""] * n, dtype=object)
    roles = (("F", manifest["train"]["subjects"]),
             ("H", manifest.get("holdout", {}).get("subjects") or []),
             ("S", manifest.get("support", {}).get("subjects") or []))
    missing = {}
    for tag, rows in roles:
        miss = 0
        for row in rows:
            name = row["subject"] if isinstance(row, dict) else row
            if name not in pos:
                miss += 1
                continue
            i = pos[name]
            if split[i] != "-":
                raise ValueError(f"subject {name!r} is in two manifest roles")
            split[i] = tag
            if isinstance(row, dict) and row.get("subject_id"):
                subject_id[i] = row["subject_id"]
        missing[tag] = miss
    if missing["F"]:
        raise ValueError(f"{missing['F']} T subjects are absent from the cache")

    cand_index = np.asarray(cache["cand_index"], dtype=np.int64)
    gold_token = cand_tok[cand_index]
    recall_token = cand_tok[zs]
    hit = recall_token == gold_token[:, None]

    t_rows = np.asarray([pos[r["subject"]] for r in manifest["train"]["subjects"]], np.int64)
    c = CANONICAL_TEMPLATE
    cand_acc = _class_macro_mean(hit[t_rows, c], gold_token[t_rows])
    summary = {
        "schema": READOUT_SCHEMA,
        "relation": cache["meta"]["relation"],
        "model": cache["meta"]["model"],
        "n_subjects": int(n),
        "n_templates": int(t),
        "n_candidates": int(len(cand_tok)),
        "split_counts": {k: int((split == k).sum()) for k in ("F", "H", "S", "-")},
        "manifest_subjects_missing_from_cache": missing,
        "canonical_template_index": int(c),
        "pre_rel_cand_acc": float(cand_acc),
        "pre_rel_cand_acc_micro": float(hit[t_rows, c].mean()),
        "acc_micro_by_split_template": {
            k: [float(hit[split == k, j].mean()) if (split == k).any() else None
                for j in range(t)]
            for k in ("F", "H")
        },
        "argmax_exact_ties": int(((arg != zs) & tie_ok).sum()),
        "definitions": {
            "recall_token": "cand_first_token[zs_cand] (effective-class token id)",
            "hit": "recall_token == cand_first_token[cand_index]",
            "pre_rel_cand_acc": "class-macro mean of hit over T (manifest train, "
                                "manifest order) at template 0, grouped by gold token",
        },
    }
    arrays = {
        "subjects": np.asarray(subjects, dtype=object).astype(str),
        "subject_id": subject_id.astype(str),
        "split": split,
        "templates": np.asarray(list(cache["rel_templates"]), dtype=object).astype(str),
        "candidates": np.asarray(list(cache["candidates"]), dtype=object).astype(str),
        "cand_first_token": cand_tok,
        "cand_index": cand_index,
        "gold_token": gold_token,
        "zs_cand": zs,
        "recall_token": recall_token,
        "hit": hit,
        "cand_logits": L,
    }
    return arrays, summary

def readout(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="prerel_recall.py readout")
    p.add_argument("--repo", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--logits", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="<name>.npz; <name>.json is the summary")
    p.add_argument("--expect-cand-acc", type=float, default=None,
                   help="the fleet row's pre_rel_cand_acc; refuse on a mismatch")
    p.add_argument("--tolerance", type=float, default=1e-12)
    p.add_argument("--cache-meta", default=None,
                   help="also write the cache's string fields (subjects, objects, "
                        "cand_objects, candidates, rel_templates) here as JSON, so a "
                        "reader of the readout never needs the .pt")
    a = p.parse_args(argv)

    import numpy as np
    from linroute.cache import read_cache

    cache_path, logits_path = Path(a.cache).resolve(), Path(a.logits).resolve()
    manifest_path = Path(a.manifest).resolve()
    cache = read_cache(cache_path)
    with np.load(logits_path, allow_pickle=False) as z:
        logits = {k: z[k] for k in z.files}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays, summary = recall_readout(cache, logits, manifest)
    summary["inputs"] = {
        "cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
        "logits": {"path": str(logits_path), "sha256": sha256_file(logits_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "cache_git_commit": cache["meta"].get("git_commit"),
    }
    if a.cache_meta:
        meta_out = {
            "schema": "linroute/prerel-cache-meta/1",
            "model": cache["meta"]["model"], "relation": cache["meta"]["relation"],
            "cache_sha256": summary["inputs"]["cache"]["sha256"],
            **{k: [list(x) if isinstance(x, (list, tuple)) else x for x in cache[k]]
               for k in ("subjects", "objects", "cand_objects", "candidates", "rel_templates")},
        }
        cm = Path(a.cache_meta).resolve()
        cm.with_name(cm.name + ".tmp").write_text(json.dumps(meta_out, ensure_ascii=False) + "\n")
        os.replace(cm.with_name(cm.name + ".tmp"), cm)
    ok = True
    if a.expect_cand_acc is not None:
        diff = abs(summary["pre_rel_cand_acc"] - a.expect_cand_acc)
        ok = diff <= a.tolerance
        summary["check_pre_rel_cand_acc"] = {
            "expected": a.expect_cand_acc, "got": summary["pre_rel_cand_acc"],
            "abs_diff": diff, "tolerance": a.tolerance, "passed": ok,
        }
    out = Path(a.out).resolve()
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    js = out.with_suffix(".json")
    js.with_name(js.name + ".tmp").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    os.replace(js.with_name(js.name + ".tmp"), js)
    word = "RECALL_OK" if ok else "RECALL_MISMATCH"
    print(f"{word} {summary['model']} / {summary['relation']}: "
          f"cand_acc={summary['pre_rel_cand_acc']!r} expect={a.expect_cand_acc!r} -> {out}",
          flush=True)
    return 0 if ok else 3

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("capture", "readout"):
        raise SystemExit("usage: prerel_recall.py {capture,readout} ...")
    use_repo(_repo_from_argv(argv[1:]))
    return capture(argv[1:]) if argv[0] == "capture" else readout(argv[1:])

if __name__ == "__main__":
    raise SystemExit(main())
