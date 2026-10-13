#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import choice_logit as cl
from linroute import failure_analysis as fa
from linroute import relation_sets

W = Path(os.environ.get("LINROUTE_WORK", "work"))
GCLOUD = os.environ.get("LINROUTE_GCLOUD", "gcloud")
GCLOUD_ENV = {}
BUCKET = {"main": os.environ.get("LINROUTE_RESULTS_MAIN", "gs://RESULTS/main"), "p4": os.environ.get("LINROUTE_RESULTS_P4", "gs://RESULTS/p4")}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_jsonl(path: Path, rows) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, path)

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]

def cmd_cells(a) -> int:
    data = json.loads(a.report_data.read_text())
    listed = {}
    for src, f in (("main", a.ls_main), ("p4", a.ls_p4)):
        for line in f.read_text().split():
            parts = line.split("/")
            host, slug, rel = parts[4], parts[6], parts[7]
            listed.setdefault((src, slug, rel), []).append("/".join(parts[:-1]))
    rows = {}
    for line in open(a.rows):
        r = json.loads(line)
        if r.get("repeat") == 0:
            rows[(r["model"], r["relation"], r.get("arm_set"), r.get("row_source"))] = r
    by = {}
    for c in data["cells"]:
        if c["fleet"] != "main" or c["repeat"] != 0 or c["arm_set"] not in ("p4", "shared"):
            continue
        prev = by.get((c["model"], c["relation"]))
        if prev is None or (prev["arm_set"] == "shared" and c["arm_set"] != "shared"):
            by[(c["model"], c["relation"])] = c
    out = []
    for (model, rel), c in sorted(by.items()):
        src = "p4" if c["row_source"] == "p4" else "main"
        dirs = listed.get((src, c["slug"], rel), [])
        if len(dirs) != 1:
            raise SystemExit(f"{model}/{rel}: {len(dirs)} run dirs under {src}: {dirs}")
        row = rows.get((model, rel, c["arm_set"], c["row_source"]))
        if row is None:
            raise SystemExit(f"{model}/{rel}: no rows.jsonl row for {c['arm_set']}/{c['row_source']}")

        arms = [W / stage / "arms" / c["slug"] / rel / "r0/arm.json" for stage in ("stage-179-p4", "stage-179")]
        match = [p for p in arms if p.exists() and sha256(p) == row["arm_sha256"]]
        if not match:
            raise SystemExit(f"{model}/{rel}: no arm file with the row's arm_sha256 among {arms}")
        arm = match[0]
        out.append({
            "model": model, "slug": c["slug"], "relation": rel, "relation_set": fa.relation_set(rel),
            "arm_set": c["arm_set"], "row_source": c["row_source"], "run_dir": dirs[0],
            "arm": str(arm), "manifest": str(W / "stage-179/ea/manifests" / c["slug"] / f"{rel}.json"),
            "ce": str(a.ce_root / c["ce_host"] / c["slug"] / rel / "lin_cls_ce.json") if c.get("ce_host") else None,
            "x": c["x"], "y": c["y"], "ce_status": c["ce_status"],
            "usable": c["x"] is not None and c["y"] is not None and c["ce_status"] == "passed",
            "n_classes": c["n_classes"], "n_heldout": c["n_heldout"],
            "pre_rel_cand_acc_micro": row.get("pre_rel_cand_acc_micro"),
            "pre_rel_cls_acc": row.get("pre_rel_cls_acc"),
        })
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} cells, {sum(c['usable'] for c in out)} usable, "
          f"{Counter(c['relation_set'] for c in out)}")
    return 0

def cmd_cells_extra(a) -> int:
    out = json.loads(a.base.read_text()) if a.base else []
    have = {(c["model"], c["relation"]) for c in out}
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    rows = [r for r in rows if r["model"].split("/")[-1].casefold() == a.model.casefold()
            and r.get("repeat", 0) == 0 and r.get("final", True)]
    arms, mans = {}, {}
    for st in a.stages:
        for p in st.glob("arms/*/*/r0/arm.json"):
            arms.setdefault(sha256(p), p)
        for p in st.glob("ea/manifests/*/*.json"):
            mans.setdefault(sha256(p), p)
    for r in sorted(rows, key=lambda r: r["relation"]):
        slug, rel, disp = a.slug, r["relation"], r["model"].split("/")[-1]
        if (disp, rel) in have:
            raise SystemExit(f"{disp}/{rel}: already in --base")
        tree = "runs-gcp-p4" if r.get("row_source") == "p4" else "runs-gcp"
        dirs = sorted((a.archive / tree).glob(f"*/runs/{slug}/{rel}/r0"))
        if len(dirs) != 1:
            raise SystemExit(f"{disp}/{rel}: {len(dirs)} run dirs under {tree}: {dirs}")
        node = dirs[0].parents[3]
        ce = node / "ce-out" / slug / rel / "lin_cls_ce.json"
        arm, man = arms.get(r["arm_sha256"]), mans.get(r["manifest_sha256"])
        if arm is None or man is None:
            raise SystemExit(f"{disp}/{rel}: no arm ({arm}) or manifest ({man}) with the row's sha256")
        x, y = r.get("pre_rel_cls_ce_gain"), r.get("heldout_cand_acc")
        out.append({
            "model": disp, "slug": slug, "relation": rel, "relation_set": fa.relation_set(rel),
            "arm_set": r.get("arm_set"), "row_source": r.get("row_source"), "run_dir": str(dirs[0]),
            "arm": str(arm), "manifest": str(man), "ce": str(ce) if ce.is_file() else None,
            "x": x, "y": y, "ce_status": "passed" if x is not None else "missing",
            "usable": x is not None and y is not None,
            "n_classes": r.get("n_classes"), "n_heldout": r.get("n_heldout"),
            "pre_rel_cand_acc_micro": r.get("pre_rel_cand_acc_micro"), "pre_rel_cls_acc": r.get("pre_rel_cls_acc"),
        })
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} cells ({len(rows)} added for {a.model}), {sum(c['usable'] for c in out)} usable")
    return 0

def pull_one(cell: dict, dest: Path) -> dict:
    out = dest / cell["slug"] / f"{cell['relation']}.jsonl"
    if out.exists():
        return {"cell": f"{cell['slug']}/{cell['relation']}", "status": "exists"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        urls = [f"{cell['run_dir']}/preds_ep{e:03d}.jsonl" for e in fa.EPOCHS]
        if cell["run_dir"].startswith("/"):
            for u in urls:
                shutil.copy(u, tmp)
        else:
            env = dict(os.environ, **GCLOUD_ENV)
            subprocess.run([GCLOUD, "storage", "cp", "-q", *urls, tmp], check=True, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows, digests = [], {}
        t_eval = fa.evaluation_template(json.loads(Path(cell["arm"]).read_text()))
        for e in fa.EPOCHS:
            p = Path(tmp) / f"preds_ep{e:03d}.jsonl"
            digests[p.name] = sha256(p)
            for line in open(p):
                r = json.loads(line)
                if r["group"] != "heldout":
                    continue
                if r["template"] != t_eval:
                    r.pop("logits", None)
                r["epoch"] = e
                rows.append(r)
        write_jsonl(out, rows)
    return {"cell": f"{cell['slug']}/{cell['relation']}", "status": "pulled", "sha256": digests}

def cmd_pull(a) -> int:
    cells = json.loads(a.cells.read_text())
    a.dest.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex, open(a.dest / "_pull.jsonl", "a") as log:
        for rec in ex.map(lambda c: pull_one(c, a.dest), cells):
            log.write(json.dumps(rec, sort_keys=True) + "\n")
            log.flush()
    print("pulled", len(cells))
    return 0

def find_cache(root: Path | None, slug: str, rel: str) -> Path | None:
    if root is None:
        return None
    hits = sorted((root / slug / rel).rglob("*.pt")) if (root / slug / rel).is_dir() else []
    if len(hits) > 1:
        raise SystemExit(f"{slug}/{rel}: {len(hits)} caches under {root}: {hits}")
    return hits[0] if hits else None

def backfill_dir(root: Path | None, slug: str, rel: str) -> Path | None:
    if root is None:
        return None
    d = root / slug / rel
    return d if (d / "DONE.json").is_file() else None

def backfill_readout(d: Path, cell: dict, gold_of: dict, f_names: list, h_names: list):
    z = np.load(d / "recall" / "prerel_recall.npz")
    meta = json.loads((d / "recall" / "cache_meta.json").read_text())
    subj = [str(x) for x in z["subjects"]]
    pos = {n: i for i, n in enumerate(subj)}
    if subj != [str(x) for x in meta["subjects"]]:
        raise ValueError("cache_meta subjects differ from prerel_recall subjects")
    gold_tok = np.asarray(z["gold_token"], dtype=np.int64)
    for n in f_names + h_names:
        if int(gold_tok[pos[n]]) != gold_of[n]:
            raise ValueError(f"{n!r}: backfill gold class differs from the arm's")
    rt = np.asarray(z["recall_token"][:, 0], dtype=np.int64)
    recall = {n: int(rt[pos[n]]) for n in f_names + h_names}
    tok = [int(t) for t in z["cand_first_token"]]
    cand_name = {t: str(nm) for t, nm in zip(tok, meta["candidates"])}
    cand_members = {t: [str(o) for o in grp] for t, grp in zip(tok, meta["cand_objects"])}
    gold_object = {n: str(meta["objects"][pos[n]]) for n in f_names + h_names}
    check = {"src_template": meta["rel_templates"][0], "backfill": str(d),
             "cache_sha256": meta.get("cache_sha256")}
    probe = None
    extras: dict[str, dict] = {}
    knn_npz = d / "knn" / "knn_readout.npz"
    if knn_npz.is_file():
        k = np.load(knn_npz)
        fct = np.asarray(k["f_class_token"], dtype=np.int64)
        for j, n in enumerate(str(x) for x in k["h_subjects"]):
            ex = extras.setdefault(n, {})
            for space in ("raw", "pca"):
                cls = [int(c) for c in fct[np.asarray(k[f"{space}_idx"][j], dtype=np.int64)]]
                ex[f"knn1_{space}"], ex[f"knn5_{space}"] = cls[0], fa.knn_vote(cls, 5)
        check["knn_layer"] = int(k["layer"])
    ce_npz = d / "ce-readout" / "ce_readout.npz"
    if ce_npz.is_file():
        e = np.load(ce_npz)
        ct = np.asarray(e["class_token"], dtype=np.int64)
        pf = dict(zip((str(x) for x in e["f_subjects"]), ct[np.asarray(e["f_oof_pred"], dtype=np.int64)]))
        ph = dict(zip((str(x) for x in e["h_subjects"]), ct[np.asarray(e["h_prob"]).argmax(1)]))
        probe = {n: int(pf[n]) for n in f_names} | {n: int(ph[n]) for n in h_names}
        hp = np.asarray(e["h_prob"])
        for j, n in enumerate(str(x) for x in e["h_subjects"]):
            extras.setdefault(n, {})["probe_order"] = [int(c) for c in ct[np.argsort(-hp[j], kind="stable")]]

        fleet = json.loads(Path(cell["ce"]).read_text())["ce"]["results"][0]
        present = np.unique(np.asarray(z["cand_index"], dtype=np.int64)[[pos[n] for n in f_names]])
        fleet_tok = np.asarray(z["cand_first_token"], dtype=np.int64)[present][np.asarray(fleet["pred"], dtype=np.int64)]
        check["probe_F_equals_fleet_oof"] = float(np.mean([probe[n] == int(t) for n, t in zip(f_names, fleet_tok)]))
        check["probe_layer"] = json.loads((d / "ce-readout" / "ce_readout.json").read_text())["checks"]["modal_layer"]
    check["_extras"] = extras
    return recall, probe, cand_name, cand_members, gold_object, check

def readout_cell(cell: dict, preds_root: Path, wild_root: Path | None, device: str,
                 backfill_root: Path | None = None):
    arm = json.loads(Path(cell["arm"]).read_text())
    manifest = json.loads(Path(cell["manifest"]).read_text())
    animal_class = fa.invert_bijection(arm["class_animal"])
    animals = list(arm["animals"])
    gold_of = {s["town"]: int(s["effective_class"]) for s in arm["subjects"]}
    split_of = {s["town"]: s["split"] for s in arm["subjects"]}
    f_names = [r["subject"] for r in manifest["train"]["subjects"]]
    h_names = [r["subject"] for r in manifest["holdout"]["subjects"]]
    if sorted(f_names) != sorted(n for n, s in split_of.items() if s == "train") or \
            sorted(h_names) != sorted(n for n, s in split_of.items() if s == "heldout"):
        raise ValueError("arm and manifest disagree on the F/H split")
    f_count = Counter(gold_of[n] for n in f_names)
    majority_f = min(c for c, k in f_count.items() if k == max(f_count.values()))

    by_epoch = defaultdict(list)
    for r in read_jsonl(preds_root / cell["slug"] / f"{cell['relation']}.jsonl"):
        by_epoch[r["epoch"]].append(r)
    t_eval = fa.evaluation_template(arm)
    post = {e: fa.heldout_preds(by_epoch[e], t_eval) for e in fa.EPOCHS}
    check = {"cell": f"{cell['slug']}/{cell['relation']}", "model": cell["model"],
             "relation": cell["relation"], "eval_template_name": t_eval,
             "eval_template": arm["templates"]["gate"][t_eval]}

    t5 = post[20]
    y_re = fa.class_macro([t5[n]["correct"] == 1 for n in h_names], [gold_of[n] for n in h_names])
    check["y_recomputed"], check["y_row"] = y_re, cell["y"]

    recall = probe = {}
    extras: dict[str, dict] = {}
    probe_meta = {}
    cand_name, cand_members, gold_object = {}, {}, {}
    bdir = backfill_dir(backfill_root, cell["slug"], cell["relation"])
    cache_path = find_cache(wild_root, cell["slug"], cell["relation"])
    if bdir is not None:
        recall, probe, cand_name, cand_members, gold_object, bchk = backfill_readout(
            bdir, cell, gold_of, f_names, h_names)
        extras = bchk.pop("_extras")
        check.update(bchk)
        check["readout_source"] = "backfill"
        check["recall_micro_F"] = float(np.mean([recall[n] == gold_of[n] for n in f_names]))
        check["recall_micro_F_row"] = cell["pre_rel_cand_acc_micro"]
        if probe is None:
            probe = {}
            check["readout_source"] = "backfill recall; probe needs the cache" if cache_path is None \
                else "backfill recall; probe from the fleet CE artifact on the cache"
    backfill_recall = dict(recall) if bdir is not None else None
    if cache_path is not None and not probe:
        from linroute.cache import read_cache
        from linroute.transfer import CANONICAL_TEMPLATE, manifest_fold_plan

        cache = read_cache(cache_path)
        pos = {n: i for i, n in enumerate(cache["subjects"])}
        f_rows = np.asarray([pos[n] for n in f_names], dtype=np.int64)
        h_rows = np.asarray([pos[n] for n in h_names], dtype=np.int64)
        tok = np.asarray(cache["cand_first_token"], dtype=np.int64)
        cidx = np.asarray(cache["cand_index"], dtype=np.int64)
        zs = np.asarray(cache["zs_cand"][:, CANONICAL_TEMPLATE], dtype=np.int64)
        for n in f_names + h_names:
            if int(tok[cidx[pos[n]]]) != gold_of[n]:
                raise ValueError(f"{n!r}: cache gold class differs from the arm's")
        recall = {n: int(tok[zs[pos[n]]]) for n in f_names + h_names}
        cand_name = {int(t): str(nm) for t, nm in zip(tok, cache["candidates"])}

        cand_members = {int(t): [str(o) for o in grp] for t, grp in zip(tok, cache["cand_objects"])}
        gold_object = {n: str(cache["objects"][pos[n]]) for n in f_names + h_names}
        check["src_template"] = cache["rel_templates"][CANONICAL_TEMPLATE]
        check["recall_micro_F"] = float(np.mean([recall[n] == gold_of[n] for n in f_names]))
        check["recall_micro_F_row"] = cell["pre_rel_cand_acc_micro"]

        ce = json.loads(Path(cell["ce"]).read_text())["ce"]["results"][0]
        present, dense_f, class_tok = fa.dense_axis(cidx[f_rows], tok)
        n_classes = len(present)
        S = np.asarray(cache["rel_subject_last"][:, CANONICAL_TEMPLATE], dtype=np.float64)
        oof = np.asarray(ce["pred"], dtype=np.int64)
        folds, _inner = manifest_fold_plan(manifest, f_names)
        sel = sorted(ce["selected_pairs"], key=lambda s: s["outer"])
        check["oof_refit"] = fa.oof_refit_check(S, f_rows, dense_f, n_classes, folds, sel, oof,
                                                device=device) if cell.get("_check_oof") else None
        layer = fa.modal_layer([p["layer"] for p in sel if p.get("available", True)])
        h_dense, probe_meta = fa.h_probe_readout(S, f_rows, h_rows, dense_f, n_classes, layer,
                                                 device=device)
        probe = {n: int(class_tok[oof[i]]) for i, n in enumerate(f_names)}
        probe.update({n: int(class_tok[h_dense[i]]) for i, n in enumerate(h_names)})
        check["probe_layer"] = layer
        check["cache"] = str(cache_path)
        check.setdefault("readout_source", "cache")
        if backfill_recall is not None:
            check["recall_equals_backfill"] = float(np.mean([recall[n] == backfill_recall[n] for n in recall]))
    if probe:
        check["probe_acc_F_oof_micro"] = float(np.mean([probe[n] == gold_of[n] for n in f_names]))
        check["probe_acc_H_micro"] = float(np.mean([probe[n] == gold_of[n] for n in h_names]))

    out = []
    retained = set(animal_class.values())
    check["class_animal"] = arm["class_animal"]
    if cell["relation"].startswith("engineered_") and cell["relation"].endswith("_band"):
        bands = {int(t): fa.band_rank(nm) for t, nm in cand_name.items()}
        if bands and None not in bands.values():
            check["class_band"] = {str(t): b for t, b in bands.items()}

    def name(c):
        if c is None:
            return None
        return arm["class_label"].get(str(c)) or cand_name.get(int(c))

    def label(c):
        return None if c is None else arm["class_animal"].get(str(c))

    for split, names in (("F", f_names), ("H", h_names)):
        for n in names:
            row = {"model": cell["model"], "relation": cell["relation"],
                   "relation_set": cell["relation_set"], "subject": n, "split": split,
                   "gold": gold_of[n], "n_classes": len(retained), "majority_f": majority_f,
                   "recall": recall.get(n), "probe": probe.get(n),
                   "recall_in_E": (recall[n] in retained) if n in recall else None}
            if split == "H":
                for e in fa.EPOCHS:
                    r = post[e].get(n)
                    row[f"pred_{e}"] = animal_class[r["pred"]] if r else None
                for ref in ("gold", "recall", "probe", "pred_20"):
                    row[f"{ref}_name"] = name(row[ref])
                    row[f"{ref}_label"] = label(row[ref])
                    if row[ref] is not None and int(row[ref]) in cand_members:
                        m = cand_members[int(row[ref])]
                        row[f"{ref}_members"], row[f"{ref}_n_members"] = m[:4], len(m)
                row["gold_object"] = gold_object.get(n)
                row.update(extras.get(n, {}))
                t5r = post[20][n]
                row["label_pred"], row["label_gold"] = t5r["pred"], t5r["target"]
                lg = np.asarray(t5r["logits"])
                row["rank_gold_20"] = int((lg > lg[animals.index(t5r["target"])]).sum())
                if n in probe:
                    row["rank_probe_20"] = int((lg > lg[animals.index(arm["class_animal"][str(probe[n])])]).sum())
            out.append(row)
    return out, check

def cmd_readout(a) -> int:
    cells = json.loads(a.cells.read_text())
    if a.only:
        cells = [c for c in cells if f"{c['slug']}/{c['relation']}" in set(a.only.split(","))]
    for c in cells:
        c["_check_oof"] = a.check_oof
    rows, checks = [], []
    one = functools.partial(readout_cell, preds_root=a.preds, wild_root=a.wild_root, device=a.device,
                            backfill_root=a.backfill_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        for c, (r, k) in zip(cells, ex.map(one, cells)):
            rows.extend(r)
            checks.append(k)
    write_jsonl(a.out, rows)
    write_jsonl(a.out.with_name("readout-checks.jsonl"), checks)
    print(f"{len(rows)} subject rows from {len(cells)} cells")
    return 0

def scores_cell(cell: dict, backfill_root: Path) -> dict | None:
    d = backfill_dir(backfill_root, cell["slug"], cell["relation"])
    if d is None:
        return None
    z = np.load(d / "recall" / "prerel_recall.npz")
    subj = [str(x) for x in z["subjects"]]
    tok = np.asarray(z["cand_first_token"], dtype=np.int64)
    L = np.asarray(z["cand_logits"][:, 0, :], dtype=np.float64)
    classes = sorted(set(int(t) for t in tok))

    S = np.stack([L[:, tok == c].max(1) for c in classes], 1)
    chk = {"recall_argmax_equals_recall_token":
           float(np.mean(tok[L.argmax(1)] == np.asarray(z["recall_token"][:, 0], dtype=np.int64)))}
    H: dict[str, dict] = {}
    ce_npz, knn_npz = d / "ce-readout" / "ce_readout.npz", d / "knn" / "knn_readout.npz"
    if ce_npz.is_file():
        e = np.load(ce_npz)
        ct = [int(c) for c in e["class_token"]]
        lp = np.log(np.clip(np.asarray(e["h_prob"], dtype=np.float64), 1e-12, None))
        for j, n in enumerate(str(x) for x in e["h_subjects"]):
            H.setdefault(n, {})["probe_logp"] = {str(c): round(float(v), 5) for c, v in zip(ct, lp[j])}
    if knn_npz.is_file():
        k = np.load(knn_npz)
        fct = np.asarray(k["f_class_token"], dtype=np.int64)
        for j, n in enumerate(str(x) for x in k["h_subjects"]):
            near = Counter(int(c) for c in fct[np.asarray(k["raw_idx"][j][:5], dtype=np.int64)])
            H.setdefault(n, {})["knn5_share"] = {str(c): v / 5 for c, v in near.items()}
    pos = {n: i for i, n in enumerate(subj)}
    for n, h in H.items():
        h["recall_score"] = {str(c): round(float(v), 4) for c, v in zip(classes, S[pos[n]])}
    chk["n_H"] = len(H)
    return {"model": cell["model"], "relation": cell["relation"], "H": H, "check": chk}

def cmd_scores(a) -> int:
    cells = [c for c in json.loads(a.cells.read_text()) if c.get("usable", True)]
    one = functools.partial(scores_cell, backfill_root=a.backfill_root)
    n = 0
    with cf.ProcessPoolExecutor(a.jobs) as ex, a.out.open("w") as fh:
        for r in ex.map(one, cells):
            if r is not None:
                fh.write(json.dumps(r) + "\n")
                n += 1
    print(f"scores for {n} of {len(cells)} cells")
    return 0

def cmd_labels(a) -> int:
    from transformers import AutoTokenizer

    cells = json.loads(a.cells.read_text())
    animals = defaultdict(set)
    slug_of = {}
    for c in cells:
        arm = json.loads(Path(c["arm"]).read_text())
        animals[c["model"]].update(arm["animals"])
        slug_of[c["model"]] = c["slug"]
    out = {"labels": {}, "tokenizers": {}}
    for model, labs in sorted(animals.items()):
        hf_id = slug_of[model].replace("__", "/", 1)
        tok = AutoTokenizer.from_pretrained(hf_id, local_files_only=True, trust_remote_code=True)
        ids = {}
        for lab in sorted(labs):
            t = tok.encode(" " + lab, add_special_tokens=False)
            ids[lab] = [int(i) for i in t if tok.decode([i]).strip()]
        out["labels"][model] = ids
        out["tokenizers"][model] = {"hf_id": hf_id, "name_or_path": str(tok.name_or_path),
                                    "class": type(tok).__name__}
        print(model, len(ids))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0

RATES = ("rate_probe", "rate_recall", "rate_both", "rate_freq", "rate_chance_probe",
         "rate_chance_recall", "rate_chance_freq", "rate_indep_probe", "rate_indep_recall",
         "rate_probe_given_ref_wrong", "rate_recall_given_ref_wrong")

QUADRANTS = {"probe+recall+": ("111", "110"), "probe+recall-": ("101", "100"),
             "probe-recall+": ("011", "010"), "probe-recall-": ("001", "000")}

def pooled(stats: list[dict]) -> dict:
    keys = {k for s in stats for k in s}
    tot = {k: sum(s.get(k, 0) for s in stats) for k in keys}
    out = dict(tot)
    ne = tot.get("n_err", 0)
    for ref in ("probe", "recall"):
        if ne and f"agree_{ref}" in tot:
            out[f"rate_{ref}"] = tot[f"agree_{ref}"] / ne
            out[f"rate_chance_{ref}"] = tot[f"chance_{ref}"] / ne
            out[f"rate_indep_{ref}"] = tot[f"indep_{ref}"] / ne
            nw = tot[f"n_err_{ref}_wrong"]
            out[f"rate_{ref}_given_ref_wrong"] = tot[f"agree_{ref}"] / nw if nw else None
    if ne and "agree_freq" in tot:
        out["rate_freq"] = tot["agree_freq"] / ne
        out["rate_chance_freq"] = tot["chance_freq"] / ne
        out["rate_both"] = tot.get("agree_both", 0) / ne
    return out

def quadrants(cube: dict) -> dict:
    out = {}
    for q, (right, wrong) in QUADRANTS.items():
        n = cube.get(right, 0) + cube.get(wrong, 0)
        out[q] = {"n": n, "ft_right": cube.get(right, 0), "p_ft_right": cube.get(right, 0) / n if n else None}
    return out

def boot_ci(per_cell: list[tuple[dict, dict]], n_boot: int, seed: int) -> dict:
    if len(per_cell) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(n_boot):
        idx = rng.integers(0, len(per_cell), len(per_cell))
        p = pooled([per_cell[i][0] for i in idx])
        cube = Counter()
        for i in idx:
            cube.update(per_cell[i][1])
        for k in RATES:
            if p.get(k) is not None:
                draws[k].append(p[k])
        for q, v in quadrants(cube).items():
            if v["p_ft_right"] is not None:
                draws["q:" + q].append(v["p_ft_right"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}

def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return None
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

def pick_examples(rows: list[dict], per_type: int) -> dict:
    types = {
        "error_is_probe_class": lambda r: r["pred_20"] != r["gold"] and r["pred_20"] == r["probe"] != r["recall"],
        "error_is_recalled_class": lambda r: r["pred_20"] != r["gold"] and r["pred_20"] == r["recall"] != r["probe"],
        "recalled_not_read_out_fails": lambda r: r["recall"] == r["gold"] != r["probe"] and r["pred_20"] != r["gold"],
        "read_out_not_recalled_succeeds": lambda r: r["probe"] == r["gold"] != r["recall"] and r["pred_20"] == r["gold"],
        "both_right_fails": lambda r: r["probe"] == r["gold"] == r["recall"] and r["pred_20"] != r["gold"],
    }
    keyed = sorted((hashlib.sha1(f"{r['model']}|{r['relation']}|{r['subject']}".encode()).hexdigest(), r)
                   for r in rows if r["relation_set"] == "Hernandez")
    out = {}
    for t, pred in types.items():
        seen, chosen = set(), []
        for _, r in keyed:
            if len(chosen) >= per_type:
                break
            if (r["model"], r["relation"]) in seen or not pred(r):
                continue
            seen.add((r["model"], r["relation"]))
            chosen.append({k: r.get(k) for k in (
                "model", "relation", "subject", "gold_object", "gold_members", "gold_n_members",
                "recall_members", "recall_n_members", "probe_members", "probe_n_members",
                "pred_20_members", "pred_20_n_members", "gold_name", "gold_label", "recall_name", "recall_label",
                "probe_name", "probe_label", "pred_20_name", "pred_20_label", "recall_in_E",
                "rank_gold_20", "rank_probe_20")})
        out[t] = chosen
    return out

def _cell_features(errs: list[dict], ctx: dict, C: int, rows: list[dict] | None = None) -> dict[str, float]:
    refs = [fa.reference_sets(r, ctx) for r in errs]
    pw = [i for i, r in enumerate(errs) if r["probe"] != r["gold"]]
    pr = [i for i, r in enumerate(errs) if r["probe"] == r["gold"]]
    f: dict[str, float] = {"n_err": len(errs), "n_pw": len(pw), "n_pr": len(pr)}
    for scope, ix in (("all", range(len(errs))), ("pw", pw), ("pr", pr)):
        e, R = [errs[i] for i in ix], [refs[i] for i in ix]
        for key in fa.HYPOTHESES:
            h = fa.hypothesis_counts(e, R, key, C)
            for q in ("n", "match", "chance", "indep"):
                f[f"h:{scope}:{key}:{q}"] = h[q]
    m = fa.mrr_counts(errs)
    for q in ("n", "rr", "chance", "indep"):
        f[f"mrr:{q}"] = m[q]
    mr = fa.mrr_counts(errs, "recall_order")
    for q in ("n", "rr", "chance", "indep"):
        f[f"mrrr:{q}"] = mr[q]

    for scope, ix in (("all", range(len(errs))), ("pr", pr), ("pw", pw)):
        e, R = [errs[i] for i in ix], [refs[i] for i in ix]
        for pre, cnt in zip(("sv", "svsh", "svch"), fa.shapley(e, R, fa.SHAPLEY_KEYS, ctx["classes"])):
            for key, n in cnt.items():
                f[f"{pre}:{scope}:{key}"] = n

    vsh, vch = fa.decompose_baselines(errs, refs, fa.VARIANT_ORDER, ctx["classes"])
    for i, r in enumerate(errs):
        b = fa.bucket_of(r["pred_20"], refs[i], fa.VARIANT_ORDER)
        f[f"dv:all:{b}"] = f.get(f"dv:all:{b}", 0) + 1
        side = "pr" if r["probe"] == r["gold"] else "pw"
        f[f"dv:{side}:{b}"] = f.get(f"dv:{side}:{b}", 0) + 1
    for key, n in vsh.items():
        f[f"dvsh:all:{key}"] = n
    for key, n in vch.items():
        f[f"dvch:all:{key}"] = n
    for side, ix in (("pr", pr), ("pw", pw)):
        ms = fa.mrr_counts([errs[i] for i in ix])
        for q in ("n", "rr", "chance", "indep"):
            f[f"mrr:{side}:{q}"] = ms[q]
        e, R = [errs[i] for i in ix], [refs[i] for i in ix]
        for key, n in fa.decompose(e, R, fa.SIDE_ORDER).items():
            f[f"ds:{side}:{key}"] = n
        sh, ch = fa.decompose_baselines(e, R, fa.SIDE_ORDER, ctx["classes"])
        for key, n in sh.items():
            f[f"dsh:{side}:{key}"] = n
        for key, n in ch.items():
            f[f"dch:{side}:{key}"] = n
    for r, n in m["ranks"].items():
        f[f"rank:{r}"] = n
    for r in errs:
        order = r.get("probe_order")
        if order and r["pred_20"] in order:
            m = len(order) - (r["gold"] in order)
            for k in range(1, m + 1):
                f[f"rankc:{k}"] = f.get(f"rankc:{k}", 0.0) + 1.0 / m
    if rows is not None:
        f["pr_expected"] = len(errs) * float(np.mean([r["probe"] == r["gold"] for r in rows]))
        for e in fa.EPOCHS[1:]:
            ee = [dict(r, pred_20=r[f"pred_{e}"]) for r in rows
                  if r.get(f"pred_{e}") is not None and r[f"pred_{e}"] != r["gold"]]
            R = [fa.reference_sets(r, ctx) for r in ee]
            f[f"ep{e}:n_err"] = len(ee)
            for key in ("probe_top1", "recall"):
                h = fa.hypothesis_counts(ee, R, key, C)
                for q in ("n", "match", "chance", "indep"):
                    f[f"h:ep{e}:{key}:{q}"] = h[q]
    return f

def attach_recall_order(rows: list[dict], classes: set) -> None:
    for r in rows:
        sc = r.get("recall_score")
        if sc:
            r["recall_order"] = sorted(classes, key=lambda c: (-sc.get(str(c), -np.inf), c))

LOGIT_COLS = ["probe_r1", "probe_r2", "probe_r3", "recall_r1", "recall_r2", "recall_r3",
              "probe_logp", "recall_logsm", "log_freq_f", "knn5_share", "epoch0"]
LOGIT_SPECS = {
    "rank": {"columns": ["probe_r1", "probe_r2", "probe_r3", "recall_r1", "recall_r2", "recall_r3",
                         "log_freq_f", "knn5_share", "epoch0"],
             "blocks": {"probe": ["probe_r1", "probe_r2", "probe_r3"], "recall": ["recall_r1", "recall_r2", "recall_r3"],
                        "frequency": ["log_freq_f"], "knn": ["knn5_share"], "epoch0": ["epoch0"]}},
    "continuous": {"columns": ["probe_logp", "recall_logsm", "log_freq_f", "knn5_share", "epoch0"],
                   "blocks": {"probe": ["probe_logp"], "recall": ["recall_logsm"], "frequency": ["log_freq_f"],
                              "knn": ["knn5_share"], "epoch0": ["epoch0"]}},
}
LOGIT_FEATURES = {
    "probe_r1": "1 if the class is the probe's top class", "probe_r2": "1 if it is the probe's 2nd class",
    "probe_r3": "1 if it is the probe's 3rd class",
    "recall_r1": "1 if it is the base model's top class (among the cell's classes)",
    "recall_r2": "1 if it is the base model's 2nd class", "recall_r3": "1 if it is the base model's 3rd class",
    "probe_logp": "the probe's log-probability of the class (per nat)",
    "recall_logsm": "the base model's log-softmax score of the class over the cell's classes (per nat)",
    "log_freq_f": "log of the class's count among the F subjects, +0.5 (per nat, i.e. per e-fold frequency)",
    "knn5_share": "share of the 5 nearest F subjects (raw subject state) with this class (0 to 1)",
    "epoch0": "1 if it is the model's answer before fine-tuning (epoch 0)",
}

def logit_rows(by_cell: dict, checks: dict, fcount: dict) -> tuple[list, dict]:
    recs, skipped = [], Counter()
    cell_ix = {k: i for i, k in enumerate(sorted(by_cell))}
    for k, rows in sorted(by_cell.items()):
        classes = sorted(int(c) for c in (checks[k].get("class_animal") or {}))
        for r in rows:
            g, a = r["gold"], r["pred_20"]
            if a == g:
                continue
            po, ro, lp = r.get("probe_order"), r.get("recall_order"), r.get("probe_logp")
            if not (po and ro and lp and r.get("knn5_share") is not None):
                skipped["missing scores"] += 1
                continue
            cands = [c for c in classes if c != g]
            if a not in cands:
                skipped["answer outside the classes"] += 1
                continue
            prank = {c: i + 1 for i, c in enumerate(po)}
            rrank = {c: i + 1 for i, c in enumerate(ro)}
            sc = np.array([r["recall_score"].get(str(c), -np.inf) for c in classes])
            lsm = dict(zip(classes, sc - (sc.max() + np.log(np.exp(sc - sc.max()).sum()))))
            lpmin = min(lp.values())
            X = np.array([[prank.get(c) == 1, prank.get(c) == 2, prank.get(c) == 3,
                           rrank.get(c) == 1, rrank.get(c) == 2, rrank.get(c) == 3,
                           lp.get(str(c), lpmin), lsm[c], np.log(fcount[k].get(c, 0) + 0.5),
                           r["knn5_share"].get(str(c), 0.0), c == r.get("pred_0")] for c in cands], dtype=np.float64)
            recs.append({"X": X, "chosen": cands.index(a), "cell": cell_ix[k], "model": k[0]})
    return recs, dict(skipped)

def logit_design(recs: list) -> cl.Design:
    sizes = np.array([len(r["X"]) for r in recs])
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    cells = np.array([r["cell"] for r in recs])
    _, cell = np.unique(cells, return_inverse=True)
    return cl.Design(np.concatenate([r["X"] for r in recs]), start,
                     start + np.array([r["chosen"] for r in recs]), cell, list(LOGIT_COLS))

_LOGIT: dict = {}

def _logit_draw(args):
    group, spec, seed = args
    D, beta = _LOGIT[(group, spec)]
    w = cl.cell_weights(D.cell, int(D.cell.max()) + 1, np.random.default_rng(seed))
    b, ll, gains = cl.drop_one(D, LOGIT_SPECS[spec]["blocks"], w, beta)
    return group, spec, b, ll, cl.null_loglik(D, w), gains

def logit_block(by_cell: dict, checks: dict, fcount: dict, n_boot: int, jobs: int) -> dict:
    import multiprocessing as mp

    recs, skipped = logit_rows(by_cell, checks, fcount)
    groups = {"pooled": recs} | {m: [r for r in recs if r["model"] == m] for m in sorted({r["model"] for r in recs})}
    out = {"features": LOGIT_FEATURES, "n_errors": len(recs), "skipped": skipped,
           "specs": {s: {"columns": v["columns"], "blocks": v["blocks"]} for s, v in LOGIT_SPECS.items()}}
    point = {}
    for g, rs in groups.items():
        Dg = logit_design(rs)
        for spec, v in LOGIT_SPECS.items():
            D = Dg.columns(v["columns"])
            b, ll, gains = cl.drop_one(D, v["blocks"])
            _LOGIT[(g, spec)] = (D, b)
            point[(g, spec)] = (b, ll, cl.null_loglik(D), gains, len(rs), int(D.cell.max()) + 1)
    tasks = [(g, spec, 1000 * i + j) for i, g in enumerate(groups) for spec in LOGIT_SPECS
             for j in range(n_boot if g == "pooled" else max(n_boot // 3, 50))]
    draws = defaultdict(list)
    with cf.ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        for g, spec, b, ll, ll0, gains in ex.map(_logit_draw, tasks, chunksize=4):
            draws[(g, spec)].append((b, ll, ll0, gains))

    def pct(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None

    def summarise(g, spec):
        cols = LOGIT_SPECS[spec]["columns"]
        b, ll, ll0, gains, n, nc = point[(g, spec)]
        dr = draws[(g, spec)]
        B = np.array([d[0] for d in dr]) if dr else np.zeros((0, len(cols)))
        r2 = [1 - d[1] / d[2] for d in dr]
        dg = {blk: [d[3][blk] / -d[2] for d in dr] for blk in gains}
        res = {"n_err": n, "n_cells": nc, "n_boot": len(dr), "ll": ll, "ll0": ll0, "mcfadden_r2": 1 - ll / ll0,
               "ci_r2": pct(r2), "beta": dict(zip(cols, map(float, b))),
               "odds_ratio": {c: float(np.exp(x)) for c, x in zip(cols, b)},
               "ci_beta": {c: pct(B[:, i]) for i, c in enumerate(cols)},
               "ci_or": {c: (lambda q: q and [float(np.exp(q[0])), float(np.exp(q[1]))])(pct(B[:, i]))
                         for i, c in enumerate(cols)},
               "drop_ll": gains, "drop_r2": {blk: x / -ll0 for blk, x in gains.items()},
               "ci_drop_r2": {blk: pct(x) for blk, x in dg.items()},
               "probe_minus_recall_r2": (gains["probe"] - gains["recall"]) / -ll0,
               "ci_probe_minus_recall_r2": pct([p - q for p, q in zip(dg["probe"], dg["recall"])])}
        return res

    for spec in LOGIT_SPECS:
        out["specs"][spec]["pooled"] = summarise("pooled", spec)
        out["specs"][spec]["by_model"] = {g: summarise(g, spec) for g in groups if g != "pooled"}
    return out

def other_errors(by_cell: dict, cells: dict, checks: dict, tokens: dict, n_boot: int) -> dict:
    feats = {}
    for k, rows in by_cell.items():
        chk = checks[k]
        ca = chk.get("class_animal") or {}
        ctx = {"classes": {int(c) for c in ca}, "class_label": {int(c): v for c, v in ca.items()},
               "class_band": {int(c): b for c, b in (chk.get("class_band") or {}).items()},
               "label_tokens": tokens.get(k[0], {})}
        attach_recall_order(rows, ctx["classes"])
        errs = [r for r in rows if r["pred_20"] != r["gold"]]
        if errs:
            feats[k] = _cell_features(errs, ctx, rows[0]["n_classes"], rows)
    names = sorted({n for f in feats.values() for n in f})
    keys = sorted(feats)
    M = np.array([[feats[k].get(n, 0.0) for n in names] for k in keys])
    col = {n: i for i, n in enumerate(names)}

    def ratio(v, num, den):
        d = v[col[den]] if den in col else 0.0
        return float(v[col[num]] / d) if num in col and d else None

    def specs():
        out = []
        for scope in ("all", "pw", "pr"):
            for key in fa.HYPOTHESES:
                for q in ("match", "chance", "indep"):
                    out.append((f"h:{scope}:{key}:{q}", f"h:{scope}:{key}:{q}", f"h:{scope}:{key}:n"))
        for q in ("rr", "chance", "indep"):
            out.append((f"mrr:{q}", f"mrr:{q}", "mrr:n"))
            out.append((f"mrrr:{q}", f"mrrr:{q}", "mrrr:n"))
        SV = fa.SHAPLEY_KEYS + ("unexplained",)
        for scope, den in (("all", "n_err"), ("pr", "n_pr"), ("pw", "n_pw")):
            for pre in ("sv", "svsh", "svch"):
                out += [(f"{pre}:{scope}:{b}", f"{pre}:{scope}:{b}", den) for b in SV]
        for side in ("pr", "pw"):
            for q in ("rr", "chance", "indep"):
                out.append((f"mrr:{side}:{q}", f"mrr:{side}:{q}", f"mrr:{side}:n"))
            for pre in ("ds", "dsh", "dch"):
                out += [(f"{pre}:{side}:{b}", f"{pre}:{side}:{b}", f"n_{side}") for b in fa.SIDE_ORDER]
            out += [(f"dv:{side}:{b}", f"dv:{side}:{b}", f"n_{side}") for b in fa.VARIANT_ORDER]
        for pre in ("dv", "dvsh", "dvch"):
            out += [(f"{pre}:all:{b}", f"{pre}:all:{b}", "n_err") for b in fa.VARIANT_ORDER]
        out += [("share_pw", "n_pw", "n_err"), ("share_pr", "n_pr", "n_err"),
                ("share_pr_expected", "pr_expected", "n_err")]
        for e in fa.EPOCHS[1:]:
            for key in ("probe_top1", "recall"):
                for q in ("match", "chance", "indep"):
                    out.append((f"h:ep{e}:{key}:{q}", f"h:ep{e}:{key}:{q}", f"h:ep{e}:{key}:n"))
        return out

    SPECS = specs()

    def group(rows_idx: list[int], seed: int) -> dict:
        v = M[rows_idx].sum(0)
        point = {name: ratio(v, num, den) for name, num, den in SPECS}
        ci = {}
        if len(rows_idx) >= 5:
            rng = np.random.default_rng(seed)
            draws = defaultdict(list)
            arr = np.asarray(rows_idx)
            for _ in range(n_boot):
                vb = M[arr[rng.integers(0, len(arr), len(arr))]].sum(0)
                for name, num, den in SPECS:
                    x = ratio(vb, num, den)
                    if x is not None:
                        draws[name].append(x)
            ci = {n: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] for n, d in draws.items()}
        return {"v": v, "point": point, "ci": ci, "n_cells": len(rows_idx)}

    def H(g, scope, key):
        n = g["v"][col[f"h:{scope}:{key}:n"]] if f"h:{scope}:{key}:n" in col else 0
        p = lambda q: g["point"].get(f"h:{scope}:{key}:{q}")
        c = lambda q: g["ci"].get(f"h:{scope}:{key}:{q}")
        return {"n_applicable": int(n), "match": int(g["v"][col[f"h:{scope}:{key}:match"]]) if n else 0,
                "rate": p("match"), "chance": p("chance"), "indep": p("indep"),
                "ci": {"rate": c("match"), "chance": c("chance"), "indep": c("indep")}, "n_cells": g["n_cells"]}

    def D(g, side, order=fa.SIDE_ORDER, pre=("ds", "dsh", "dch")):
        den = "n_err" if side == "all" else f"n_{side}"
        n = int(g["v"][col[den]])
        get = lambda p_, b: g["point"].get(f"{p_}:{side}:{b}")
        cnt = {b: int(g["v"][col[f"{pre[0]}:{side}:{b}"]]) if f"{pre[0]}:{side}:{b}" in col else 0 for b in order}
        out = {"n_err": n, "counts": cnt, "share": {b: get(pre[0], b) for b in order},
               "ci": {b: g["ci"].get(f"{pre[0]}:{side}:{b}") for b in order}, "n_cells": g["n_cells"]}
        if pre[1]:
            out |= {"shuffle": {b: get(pre[1], b) for b in order},
                    "shuffle_ci": {b: g["ci"].get(f"{pre[1]}:{side}:{b}") for b in order},
                    "chance": {b: get(pre[2], b) for b in order}}
        return out

    def MRR(g, side=None, p=None):
        p = p or ("mrr" if side is None else f"mrr:{side}")
        return {"n": int(g["v"][col[f"{p}:n"]]) if f"{p}:n" in col else 0, "mrr": g["point"].get(f"{p}:rr"),
                "chance": g["point"].get(f"{p}:chance"), "indep": g["point"].get(f"{p}:indep"),
                "ci": {"mrr": g["ci"].get(f"{p}:rr"), "chance": g["ci"].get(f"{p}:chance"),
                       "indep": g["ci"].get(f"{p}:indep")}, "n_cells": g["n_cells"]}

    allg = group(list(range(len(keys))), 11)
    by = {"model": defaultdict(list), "set": defaultdict(list), "model_set": defaultdict(list)}
    for i, k in enumerate(keys):
        by["model"][k[0]].append(i)
        by["set"][cells[k]["relation_set"]].append(i)
        by["model_set"][f"{k[0]}|{cells[k]['relation_set']}"].append(i)
    groups = {kind: {name: group(ix, 13) for name, ix in sorted(d.items())} for kind, d in by.items()}

    hyp = {}
    for key, (label, definition) in fa.HYPOTHESES.items():
        hyp[key] = {"label": label, "definition": definition, "pooled": H(allg, "all", key),
                    "pooled_probe_wrong": H(allg, "pw", key), "pooled_probe_right": H(allg, "pr", key),
                    "by_model": {m: H(g, "all", key) for m, g in groups["model"].items()},
                    "by_set": {s_: H(g, "all", key) for s_, g in groups["set"].items()},
                    "by_model_set": {ms: H(g, "all", key) for ms, g in groups["model_set"].items()}}
    sides = {"order": list(fa.SIDE_ORDER), "labels": {b: fa.BUCKET_LABELS[b] for b in fa.SIDE_ORDER},
             "definitions": {b: fa.HYPOTHESES[b][1] for b in fa.SIDE_ORDER if b in fa.HYPOTHESES}}
    for side, name in (("pr", "probe_right"), ("pw", "probe_wrong")):
        sides[name] = {"pooled": D(allg, side),
                       "by_model": {m: D(g, side) for m, g in groups["model"].items()},
                       "by_set": {s_: D(g, side) for s_, g in groups["set"].items()}}
    V = dict(order=fa.VARIANT_ORDER, pre=("dv", "dvsh", "dvch"))
    variant = {"order": list(fa.VARIANT_ORDER), "labels": {b: fa.BUCKET_LABELS[b] for b in fa.VARIANT_ORDER},
               "definitions": {b: fa.HYPOTHESES[b][1] for b in fa.VARIANT_ORDER if b in fa.HYPOTHESES},
               "knn_rule": "knn5_raw: majority class of the 5 nearest F subjects, cosine on raw subject-last "
                           "states centered on the F mean, at the CE probe's modal layer",
               "pooled": D(allg, "all", **V),
               "by_model": {m: D(g, "all", **V) for m, g in groups["model"].items()},
               "by_set": {s_: D(g, "all", **V) for s_, g in groups["set"].items()},
               "by_side": {name: D(allg, side, fa.VARIANT_ORDER, ("dv", None, None))
                           for side, name in (("pr", "probe_right"), ("pw", "probe_wrong"))}}
    SV = dict(order=fa.SHAPLEY_KEYS + ("unexplained",), pre=("sv", "svsh", "svch"))
    shap = {"order": list(SV["order"]), "labels": {b: fa.BUCKET_LABELS[b] for b in SV["order"]},
            "definitions": {b: fa.HYPOTHESES[b][1] for b in fa.SHAPLEY_KEYS},
            "rule": "each error's credit is split equally among the explanations whose class is its answer; "
                    "this equals each explanation's marginal share averaged over all orders",
            "pooled": D(allg, "all", **SV),
            "by_model": {m: D(g, "all", **SV) for m, g in groups["model"].items()},
            "by_set": {s_: D(g, "all", **SV) for s_, g in groups["set"].items()},
            "by_side": {name: D(allg, side, **SV) for side, name in (("pr", "probe_right"), ("pw", "probe_wrong"))}}

    def PVR(g):
        return {"probe": {"top1": H(g, "all", "probe_top1"), "top3": H(g, "all", "probe_top3"), "mrr": MRR(g)},
                "recall": {"top1": H(g, "all", "recall_rank1"), "top3": H(g, "all", "recall_top3"),
                           "mrr": MRR(g, p="mrrr")}}

    pvr = {"pooled": PVR(allg), "by_model": {m: PVR(g) for m, g in groups["model"].items()},
           "by_set": {s_: PVR(g) for s_, g in groups["set"].items()}}
    ranks = {n.split(":")[1]: int(allg["v"][col[n]]) for n in names if n.startswith("rank:")}
    ranks_chance = {n.split(":")[1]: float(allg["v"][col[n]]) for n in names if n.startswith("rankc:")}
    epochs = {str(e): {"n_err": int(allg["v"][col[f"ep{e}:n_err"]]),
                       "probe_top1": H(allg, f"ep{e}", "probe_top1"), "recall": H(allg, f"ep{e}", "recall")}
              for e in fa.EPOCHS[1:] if f"ep{e}:n_err" in col}
    v = allg["v"]
    ceiling = {"n_err": int(v[col["n_err"]]), "n_err_probe_wrong": int(v[col["n_pw"]]),
               "n_err_probe_right": int(v[col["n_pr"]]), "share_probe_wrong": allg["point"]["share_pw"],
               "share_probe_right": allg["point"]["share_pr"],
               "share_probe_right_if_independent": allg["point"].get("share_pr_expected"),
               "ci": {"share_probe_wrong": allg["ci"].get("share_pw"), "share_probe_right": allg["ci"].get("share_pr"),
                      "share_probe_right_if_independent": allg["ci"].get("share_pr_expected")}}
    per_cell = []
    for k in keys:
        f = feats[k]
        if f["n_err"] >= 5:
            per_cell.append({"model": k[0], "relation": k[1], "relation_set": cells[k]["relation_set"],
                             "x": cells[k]["x"], "y": cells[k]["y"], "n_err": int(f["n_err"]),
                             "share_probe_right": f["n_pr"] / f["n_err"],
                             "share_probe_right_if_independent": f["pr_expected"] / f["n_err"]})
    pr_cells = {"per_cell": per_cell,
                "rho_x": spearman([p["x"] for p in per_cell], [p["share_probe_right"] for p in per_cell]),
                "by_model": {m: {"share": g["point"]["share_pr"], "ci": g["ci"].get("share_pr"),
                                 "if_independent": g["point"].get("share_pr_expected"),
                                 "n_err": int(g["v"][col["n_err"]]), "n_cells": g["n_cells"]}
                             for m, g in groups["model"].items()},
                "by_set": {s_: {"share": g["point"]["share_pr"], "ci": g["ci"].get("share_pr"),
                                "if_independent": g["point"].get("share_pr_expected"),
                                "n_err": int(g["v"][col["n_err"]]), "n_cells": g["n_cells"]}
                           for s_, g in groups["set"].items()}}
    return {"hypotheses": hyp, "decomposition_by_side": sides, "decomposition_variant": variant, "shapley": shap,
            "probe_vs_recall": pvr, "ceiling": ceiling,
            "mrr": {"pooled": MRR(allg), "by_model": {m: MRR(g) for m, g in groups["model"].items()},
                    "by_side": {"probe_right": MRR(allg, "pr"), "probe_wrong": MRR(allg, "pw")},
                    "rank_hist": dict(sorted(ranks.items(), key=lambda kv: int(kv[0]))),
                    "rank_hist_chance": dict(sorted(ranks_chance.items(), key=lambda kv: int(kv[0])))},
            "epochs_hyp": epochs,
            "probe_right_cells": pr_cells}

def cmd_aggregate(a) -> int:
    cells = {(c["model"], c["relation"]): c for c in json.loads(a.cells.read_text())}
    checks = {(k["model"], k["relation"]): k for k in read_jsonl(a.per_subject.with_name("readout-checks.jsonl"))}
    rows = read_jsonl(a.per_subject)
    usable = {k for k, c in cells.items() if c["usable"]}
    H = [r for r in rows if r["split"] == "H" and (r["model"], r["relation"]) in usable]
    by_cell = defaultdict(list)
    for r in H:
        if r["probe"] is not None and r["recall"] is not None:
            by_cell[(r["model"], r["relation"])].append(r)
    if a.scores:
        sc = {}
        with a.scores.open() as fh:
            for line in fh:
                d = json.loads(line)
                sc[(d["model"], d["relation"])] = d["H"]
        for r in H:
            r.update(sc.get((r["model"], r["relation"]), {}).get(r["subject"], {}))
    fcount = defaultdict(Counter)
    for r in rows:
        if r["split"] == "F":
            fcount[(r["model"], r["relation"])][r["gold"]] += 1
    ft_only = defaultdict(list)
    for r in H:
        ft_only[(r["model"], r["relation"])].append(r)

    def block(keys, seed):
        per = [(fa.cell_error_stats(by_cell[k]), fa.cube(by_cell[k])) for k in keys]
        cube = Counter()
        for _, c in per:
            cube.update(c)
        out = pooled([p for p, _ in per]) | {"n_cells": len(keys), "cube": dict(cube),
                                              "quadrants": quadrants(cube)}
        out["ci"] = boot_ci(per, a.n_boot, seed) if len(keys) >= 5 else {}
        return out

    summary = {"n_cells_panel": len(cells), "n_cells_usable": len(usable),
               "n_cells_with_readout": len(by_cell), "n_heldout_subjects": len(H),
               "n_heldout_with_readout": sum(len(v) for v in by_cell.values()),
               "readout_cells": sorted(f"{m}/{r}" for m, r in by_cell)}
    headline = set(relation_sets.named("headline"))
    for view, keep in (("all", lambda k: k[1] in headline),):
        keys = sorted(k for k in by_cell if keep(k))
        v = {"pooled": block(keys, 1) if keys else {}}
        groups = defaultdict(list)
        for k in keys:
            groups[("model", k[0])].append(k)
            groups[("set", cells[k]["relation_set"])].append(k)
            groups[("model_set", f"{k[0]}|{cells[k]['relation_set']}")].append(k)
        for kind in ("model", "set", "model_set"):
            v[f"by_{kind}"] = {name: block(ks, 7) for (kd, name), ks in sorted(groups.items()) if kd == kind}
        v["epochs"] = {}
        for e in fa.EPOCHS[1:]:
            per = [fa.cell_error_stats(by_cell[k], e) for k in keys]
            p = pooled(per)
            p["acc_micro"] = float(np.mean([r[f"pred_{e}"] == r["gold"] for k in keys for r in by_cell[k]])) if keys else None
            v["epochs"][e] = p
        per_cell = []
        for k in keys:
            s = fa.cell_error_stats(by_cell[k])
            if s.get("n_err", 0) >= 5:
                per_cell.append({"model": k[0], "relation": k[1], "relation_set": cells[k]["relation_set"],
                                 "x": cells[k]["x"], "y": cells[k]["y"], "n_err": s["n_err"],
                                 "rate_probe": s["agree_probe"] / s["n_err"],
                                 "excess_probe": (s["agree_probe"] - s["chance_probe"]) / s["n_err"],
                                 "excess_recall": (s["agree_recall"] - s["chance_recall"]) / s["n_err"]})
        v["per_cell"] = per_cell
        v["rho_x_excess_probe"] = spearman([p["x"] for p in per_cell], [p["excess_probe"] for p in per_cell])
        v["rho_x_excess_recall"] = spearman([p["x"] for p in per_cell], [p["excess_recall"] for p in per_cell])
        v["rho_x_excess_probe_by_model"] = {
            m: spearman([p["x"] for p in per_cell if p["model"] == m],
                        [p["excess_probe"] for p in per_cell if p["model"] == m])
            for m in sorted({p["model"] for p in per_cell})}
        summary[view] = v

    conc = []
    for k, rs in ft_only.items():
        err = [r for r in rs if r["pred_20"] != r["gold"]]
        if err:
            top = Counter(r["pred_20"] for r in err).most_common(1)[0][1]
            conc.append({"model": k[0], "relation": k[1], "n_err": len(err), "top_share": top / len(err),
                         "freq_share": sum(r["pred_20"] == r["majority_f"] for r in err) / len(err)})
    summary["error_concentration"] = conc
    tokens = json.loads(a.label_tokens.read_text())["labels"] if a.label_tokens else {}
    summary["all"].update(other_errors(by_cell, cells, checks, tokens, a.n_boot))
    if a.scores and a.logit_boot >= 0:
        summary["all"]["choice_logit"] = logit_block(by_cell, checks, fcount, a.logit_boot, a.jobs)
    summary["examples"] = pick_examples([r for k in by_cell for r in by_cell[k]], a.per_type)
    summary["prompts"] = {f"{m}/{r}": {"src": checks[(m, r)].get("src_template"),
                                       "eval": checks[(m, r)].get("eval_template")}
                          for m, r in by_cell}
    summary["checks"] = [checks[k] for k in sorted(by_cell)]
    a.out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    p = summary["all"]["pooled"]
    print({k: p.get(k) for k in RATES}, p.get("quadrants"))
    return 0

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cells")
    s.add_argument("--report-data", type=Path, default=W / "fleet-report/report-data.json")
    s.add_argument("--rows", type=Path, default=W / "fleet-report/main-collect/rows.jsonl")
    s.add_argument("--ce-root", type=Path, default=W / "fleet-report/main-collect/ce-out")
    s.add_argument("--ls-main", type=Path, required=True)
    s.add_argument("--ls-p4", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("cells-extra")
    s.add_argument("--rows", type=Path, required=True, help="the canonical rows (rows-joined-11m.jsonl)")
    s.add_argument("--model", required=True, help="display name, e.g. Qwen3.8-27B")
    s.add_argument("--slug", required=True, help="the fleet's model slug, e.g. Qwen__Qwen3.8-27B")
    s.add_argument("--archive", type=Path, required=True, help="the model's archive (runs-gcp/, runs-gcp-p4/)")
    s.add_argument("--stages", type=Path, nargs="+", required=True, help="dirs holding arms/ and ea/manifests/")
    s.add_argument("--base", type=Path, default=None, help="cells.json to append to")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("pull")
    s.add_argument("--cells", type=Path, required=True)
    s.add_argument("--dest", type=Path, required=True)
    s.add_argument("--jobs", type=int, default=16)
    s = sub.add_parser("readout")
    s.add_argument("--cells", type=Path, required=True)
    s.add_argument("--preds", type=Path, required=True)
    s.add_argument("--wild-root", type=Path, default=None,
                   help="<root>/<slug>/<relation>/**/<one>.pt, fleet wild-cache format")
    s.add_argument("--backfill-root", type=Path, default=None,
                   help="<root>/<slug>/<relation>/{DONE.json,recall/,ce-readout/}, prerel-backfill layout")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--only", default=None, help="comma-separated slug/relation cells")
    s.add_argument("--check-oof", action="store_true", help="refit every outer fold and compare with the artifact")
    s.add_argument("--device", default="cpu")
    s.add_argument("--jobs", type=int, default=8)
    s = sub.add_parser("aggregate")
    s.add_argument("--cells", type=Path, required=True)
    s.add_argument("--per-subject", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--per-type", type=int, default=6)
    s.add_argument("--label-tokens", type=Path, default=None, help="output of `labels` (surface similarity)")
    s.add_argument("--scores", type=Path, default=None, help="output of `scores` (recall ranks, the logit)")
    s.add_argument("--logit-boot", type=int, default=1000, help="cell bootstrap draws for the logit; -1 skips it")
    s.add_argument("--jobs", type=int, default=8)
    s = sub.add_parser("scores")
    s.add_argument("--cells", type=Path, required=True)
    s.add_argument("--backfill-root", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--jobs", type=int, default=8)
    s = sub.add_parser("labels")
    s.add_argument("--cells", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    return {"cells": cmd_cells, "pull": cmd_pull, "readout": cmd_readout, "aggregate": cmd_aggregate,
            "labels": cmd_labels, "scores": cmd_scores,
            "cells-extra": cmd_cells_extra}[a.cmd](a)

if __name__ == "__main__":
    raise SystemExit(main())
