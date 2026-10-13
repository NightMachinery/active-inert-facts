#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
import tarfile
from collections import OrderedDict, defaultdict
from pathlib import Path, PurePosixPath

VERSION = "1.1.0"
SCRIPT_PATH = "scripts/materialize_newrels.py"
ARM_FILES = ("arm.json", "build-provenance.json", "mapping.tsv",
             "train.jsonl", "gate.jsonl", "heldout.jsonl")
SPACE_MARK = "␣"
REPO_ROOT = Path(__file__).resolve().parent.parent

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: Path, md5: bool = False):
    h = hashlib.sha256()
    m = hashlib.md5() if md5 else None
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            if m:
                m.update(chunk)
    return (h.hexdigest(), m.hexdigest()) if md5 else h.hexdigest()

def jdump(o) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(", ", ": "))

def canon(o) -> bytes:
    return json.dumps(o, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")

def visible(s: str | None) -> str | None:
    return None if s is None else s.replace(" ", SPACE_MARK)

def strip_comment_lines(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("//"))

class Fail(Exception):
    pass

def need(cond, msg):
    if not cond:
        raise Fail(msg)

class Source:
    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        self.files: dict[str, bytes] = {}
        p = Path(path)
        if p.is_file():
            self.kind = "tar"
            self.sha256, self.md5 = sha256_file(p, md5=True)
            self.size = p.stat().st_size
            with tarfile.open(p, "r:") as tf:
                for m in tf:
                    if m.isfile() and self._wanted(m.name):
                        self.files[self._norm(m.name)] = tf.extractfile(m).read()
        elif p.is_dir():
            self.kind = "dir"
            self.sha256 = self.md5 = None
            self.size = None
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(p).as_posix()
                if f.is_file() and self._wanted(rel):
                    self.files[rel] = f.read_bytes()
        else:
            raise Fail(f"--arms {path}: neither a tar nor a directory")

    @staticmethod
    def _norm(name: str) -> str:
        return str(PurePosixPath(name.lstrip("./")))

    @classmethod
    def _wanted(cls, name: str) -> bool:
        n = cls._norm(name)
        return (n.startswith("arms/") or n.startswith("ea/hernandez/")
                or n.startswith("ea/pack/") or n.startswith("ea/manifests/"))

    def cells(self):
        out = set()
        for n in self.files:
            parts = n.split("/")
            if len(parts) == 5 and parts[0] == "arms" and parts[4] == "arm.json":
                out.add((parts[1], parts[2], parts[3]))
        return sorted(out)

class Tokenizers:
    def __init__(self):
        self.cache = {}

    def get(self, hf_id: str, revision: str | None):
        key = (hf_id, revision)
        if key not in self.cache:
            import transformers
            kw = dict(local_files_only=True)
            try:
                tok = transformers.AutoTokenizer.from_pretrained(hf_id, revision=revision, **kw)
                used = revision
            except Exception as e:
                raise Fail(f"tokenizer {hf_id}@{revision} not loadable offline: {e}")
            need(getattr(tok, "is_fast", False), f"{hf_id}: not a fast tokenizer")
            self.cache[key] = (tok, used)
        return self.cache[key]

def relation_catalog(src_files: dict[str, bytes]) -> dict[str, dict]:
    cat = {}
    for n, b in sorted(src_files.items()):
        if not (n.startswith("ea/hernandez/") or n.startswith("ea/pack/")) or not n.endswith(".json"):
            continue
        try:
            d = json.loads(b)
        except Exception:
            continue
        if not isinstance(d, dict) or "name" not in d:
            continue
        props = d.get("properties") or {}
        src = props.get("source") or {}
        if n.startswith("ea/pack/"):
            origin = ("BEAR-big (Wiland, Ploner and Akbik 2024), Wikidata "
                      + str(src.get("relation_id", "")) + ", CC BY-SA 4.0").strip()
        else:
            origin = "Hernandez et al. 2024 relation dataset"
        cat[d["name"].lower()] = {
            "pre_rel_name": d["name"],
            "pre_rel_prompt_templates": list(d.get("prompt_templates") or []),
            "domain": props.get("domain_name"),
            "range": props.get("range_name"),
            "source": origin,
            "source_file": n,
            "n_pairs_in_source": len(d.get("samples") or []),
        }
    return cat

LOCAL_RELATIONS = {
    "sandbox_town_region": ("the town -> region sandbox: real towns and the region each lies in "
                            "(linroute docs/sandbox.md, data/sandbox/)", "town", "region"),
    "engineered_town_country_language": ("linroute engineered relation (docs/engineered.md): Wikidata P17 "
                                         "then P37, the official language of the town's country, for "
                                         "countries with exactly one", "town", "language"),
    "engineered_town_elevation_band": ("linroute engineered relation (docs/engineered.md): Wikidata P2044 "
                                       "elevation, five quantile bands", "town", "elevation band"),
    "engineered_town_latitude_band": ("linroute engineered relation (docs/engineered.md): WGS84 latitude, "
                                      "approximate quantile bands", "town", "latitude band"),
    "engineered_town_longitude_band": ("linroute engineered relation (docs/engineered.md): WGS84 longitude, "
                                       "five quantile bands", "town", "longitude band"),
    "engineered_town_population_band": ("linroute engineered relation (docs/engineered.md): Wikidata P1082 "
                                        "population, five quantile bands", "town", "population band"),
}

REL_ALIASES = {
    "plays pro sport": "person plays pro sport",
    "person sport position": "person plays position in sport",
    "star constellation name": "star constellation",
}

def load_records(path: str, sha: str, cache: dict):
    if path not in cache:
        p = Path(path)
        need(p.is_file(), f"records file missing: {path}")
        b = p.read_bytes()
        need(sha256_bytes(b) == sha, f"records sha256 mismatch: {path}")
        cache[path] = [json.loads(l) for l in b.decode("utf-8").splitlines() if l.strip()]
    return cache[path]

def jsonl(b: bytes):
    return [json.loads(l) for l in b.decode("utf-8").splitlines() if l.strip()]

def prefix_flags(words: dict, firsts: dict) -> dict:
    out = {}
    for i, ft in firsts.items():
        core = ft.lstrip(" ")
        if not core:
            continue
        hits = sorted(w for j, ws in words.items() if j != i for w in ws
                      if w.startswith(core))
        if hits:
            out[i] = hits
    return out

def build_cell(model_slug, rel, rdir, src: Source, toks: Tokenizers, rec_cache,
               catalog, arm_set_id):
    base = f"arms/{model_slug}/{rel}/{rdir}"
    raw = {f: src.files.get(f"{base}/{f}") for f in ARM_FILES}
    for f, b in raw.items():
        need(b is not None, f"{base}/{f} missing from {src.path}")
    arm = json.loads(raw["arm.json"])
    prov = json.loads(raw["build-provenance.json"])
    rows = {k: jsonl(raw[f"{k}.jsonl"]) for k in ("train", "gate", "heldout")}
    mapping = [dict(zip(raw["mapping.tsv"].decode().splitlines()[0].split("\t"), l.split("\t")))
               for l in raw["mapping.tsv"].decode().splitlines()[1:] if l]

    repeat = int(arm["repeat"])
    need(rdir == f"r{repeat}", f"{base}: dir {rdir} vs repeat {repeat}")
    need(arm["relation_slug"] == rel, f"{base}: relation_slug {arm['relation_slug']}")
    need(arm.get("arm_kind") == "newrel", f"{base}: arm_kind {arm.get('arm_kind')}")
    model = prov["model"]
    need(model.replace("/", "__") == model_slug, f"{base}: provenance model {model}")

    pool = arm["animal_pool"]["pool"]
    tok, tok_rev = toks.get(pool["tokenizer"], pool.get("revision"))
    need(pool["tokenizer"] == model or arm["tokenizer_of_manifest"] in (model, pool["tokenizer"]),
         f"{base}: tokenizer {pool['tokenizer']} for model {model}")

    mkey = f"ea/manifests/{Path(arm['manifest']['path']).parent.name}/{Path(arm['manifest']['path']).name}"
    mbytes = src.files.get(mkey)
    need(mbytes is not None, f"{base}: manifest {mkey} not in source")
    need(sha256_bytes(mbytes) == arm["manifest"]["sha256"], f"{base}: manifest sha mismatch")
    need(prov["manifest"]["sha256"] == arm["manifest"]["sha256"], f"{base}: provenance manifest sha")
    manifest = json.loads(mbytes)

    records = load_records(arm["records"]["path"], arm["records"]["sha256"], rec_cache)
    by_sid = {}
    members = defaultdict(list)
    rep = {}
    for r in records:
        by_sid.setdefault(r["subject_id"], r)
        if r.get("status") == "included" and r.get("effective_class") is not None:
            c = int(r["effective_class"])
            if r["object"] not in members[c]:
                members[c].append(r["object"])
            rep.setdefault(c, r)

    classes = sorted(int(c) for c in arm["classes"])
    c_label = {int(k): v for k, v in arm["class_label"].items()}
    c_animal = {int(k): v for k, v in arm["class_animal"].items()}
    need(set(c_label) == set(classes) == set(c_animal), f"{base}: class key sets differ")
    need(len(set(c_animal.values())) == len(classes), f"{base}: animals not distinct")
    need(sorted(c_animal.values()) == sorted(arm["animals"]), f"{base}: animals != bijection image")

    first_tok = pool["first_token"]
    n_tok = pool["n_tokens"]
    pre_first, new_first, new_id = {}, {}, {}
    pre_ntok, pre_skip = {}, {}
    for c in classes:
        dec = tok.decode([c])
        r = rep.get(c)
        need(r is not None, f"{base}: no included record for class {c}")
        need(r["object"] == c_label[c], f"{base}: representative {r['object']!r} vs {c_label[c]!r}")
        need(int(r["scored_token_id"]) == c, f"{base}: record scored id {r['scored_token_id']} vs {c}")
        need(r["scored_token_str"] == dec, f"{base}: decode({c})={dec!r} vs record {r['scored_token_str']!r}")
        ids = tok(" " + c_label[c], add_special_tokens=False)["input_ids"]
        need(ids == list(r["isolated_token_ids"]), f"{base}: isolated ids of {c_label[c]!r} differ from record")
        need(ids[int(r["scored_token_index"])] == c, f"{base}: scored index mismatch for {c_label[c]!r}")
        pre_first[c] = dec
        pre_ntok[c] = int(r["n_object_tokens"])
        pre_skip[c] = r.get("skipped_prefix_str") or ""
    for c in classes:
        a = c_animal[c]
        ids = tok(pool["target_format"].format(animal=a), add_special_tokens=False)["input_ids"]
        need(int(first_tok[a]) == ids[0], f"{base}: first token of {a!r} {ids[0]} vs recorded {first_tok[a]}")
        need(int(n_tok[a]) == len(ids), f"{base}: n_tokens of {a!r}")
        new_id[c] = ids[0]
        new_first[c] = tok.decode([ids[0]])
    need(len(set(new_id.values())) == len(classes), f"{base}: NEW-REL first tokens collide")

    subj = []
    split_of = {}
    for s in arm["subjects"]:
        c = int(s["effective_class"])
        need(c in c_animal, f"{base}: subject class {c} not retained")
        need(s["split"] in ("train", "heldout"), f"{base}: split {s['split']}")
        need(s["town"] not in split_of, f"{base}: duplicate subject {s['town']!r}")
        split_of[s["town"]] = s["split"]
        r = by_sid.get(s["subject_id"])
        need(r is not None and r["subject"] == s["town"], f"{base}: records lack subject {s['town']!r}")
        line = OrderedDict([("subject", s["town"]), ("class", c), ("pre_rel", c_label[c]),
                            ("new_rel", c_animal[c]), ("split", s["split"])])
        if r["object"] != c_label[c]:
            line["pre_rel_object"] = r["object"]
        subj.append(line)
    for s in arm["support_subjects"]:
        c = int(s["effective_class"])
        need(s["subject"] not in split_of, f"{base}: support subject also in T/H")
        r = by_sid.get(s["subject_id"])
        need(r is not None and r["subject"] == s["subject"], f"{base}: records lack support {s['subject']!r}")
        split_of[s["subject"]] = "support"
        line = OrderedDict([("subject", s["subject"]), ("class", c),
                            ("pre_rel", rep[c]["object"] if c in rep else r["object"]),
                            ("new_rel", None), ("split", "support")])
        if c in rep and r["object"] != rep[c]["object"]:
            line["pre_rel_object"] = r["object"]
        subj.append(line)

    need(len(mapping) == len(arm["subjects"]), f"{base}: mapping rows {len(mapping)}")
    for m, s in zip(mapping, arm["subjects"]):
        c = int(s["effective_class"])
        need(m["town"] == s["town"] and m["region"] == c_label[c] and m["animal"] == c_animal[c]
             and m["heldout"] == ("1" if s["split"] == "heldout" else "0"),
             f"{base}: mapping.tsv disagrees with arm.json at {s['town']!r}")
    tmpl = {**arm["templates"]["train"], **arm["templates"]["gate"]}
    town_class = {s["town"]: int(s["effective_class"]) for s in arm["subjects"]}
    for kind, rr in rows.items():
        for r in rr:
            need(r["split"] == kind, f"{base}: {kind}.jsonl row split {r['split']}")
            c = town_class.get(r["town"])
            need(c is not None, f"{base}: {kind}.jsonl subject {r['town']!r} not in arm")
            want_split = "heldout" if kind == "heldout" else "train"
            need(split_of[r["town"]] == want_split, f"{base}: {r['town']!r} in {kind}.jsonl")
            need(r["animal"] == c_animal[c] and r["region"] == c_label[c], f"{base}: row labels")
            need(r["target"] == pool["target_format"].format(animal=c_animal[c]), f"{base}: target")
            need(r["prompt"] == tmpl[r["template"]].replace("{}", r["town"]), f"{base}: prompt render")
    n_t = sum(1 for s in arm["subjects"] if s["split"] == "train")
    n_h = sum(1 for s in arm["subjects"] if s["split"] == "heldout")
    need(len(rows["train"]) == 4 * n_t and len(rows["gate"]) == n_t
         and len(rows["heldout"]) == 5 * n_h, f"{base}: row counts")
    need(arm["n_train"] == len(rows["train"]) and arm["n_gate"] == len(rows["gate"])
         and arm["n_heldout"] == len(rows["heldout"]), f"{base}: arm.json n_* fields")

    first_train = next(s["town"] for s in arm["subjects"] if s["split"] == "train")
    tlist = []
    for role, key, file_ in (("train", "train", "train"), ("gate", "gate", "gate")):
        for name, t in arm["templates"][key].items():
            ex = next(r for r in rows[file_] if r["template"] == name and r["town"] == first_train)
            tlist.append(OrderedDict([("name", name), ("role", role), ("template", t),
                                      ("example_prompt", ex["prompt"]), ("example_target", ex["target"])]))

    cls_lines = []
    for c in classes:
        n_tr = sum(1 for s in arm["subjects"] if int(s["effective_class"]) == c and s["split"] == "train")
        n_he = sum(1 for s in arm["subjects"] if int(s["effective_class"]) == c and s["split"] == "heldout")
        line = OrderedDict([
            ("class", c), ("pre_rel", c_label[c]), ("pre_rel_first_token", visible(pre_first[c])),
            ("pre_rel_n_tokens", pre_ntok[c]),
            ("new_rel", c_animal[c]), ("new_rel_token_id", new_id[c]),
            ("new_rel_first_token", visible(new_first[c])), ("new_rel_n_tokens", int(n_tok[c_animal[c]])),
            ("n_train", n_tr), ("n_heldout", n_he)])
        if pre_skip[c]:
            line["pre_rel_skipped_prefix"] = visible(pre_skip[c])
        if len(members[c]) > 1:
            line["pre_rel_members"] = members[c]
        cls_lines.append(line)

    excl = []
    for m in manifest["classes"]:
        if m.get("retained"):
            continue
        c = int(m["effective_class"])
        excl.append(OrderedDict([("class", c), ("pre_rel", rep[c]["object"] if c in rep else None),
                                 ("pre_rel_first_token", visible(tok.decode([c]))),
                                 ("reason", m.get("reason")), ("support", m.get("support"))]))
    excl.sort(key=lambda d: d["class"])

    pre_words = {c: [c_label[c]] + [w for w in members[c] if w != c_label[c]] for c in classes}
    pf = prefix_flags(pre_words, pre_first)
    nf = prefix_flags({c: [c_animal[c]] for c in classes}, new_first)
    flags_pre = [OrderedDict([("class", c), ("pre_rel", c_label[c]),
                              ("first_token", visible(pre_first[c])),
                              ("prefix_of_other_classes_words", pf.get(c, [])),
                              ("n_merged_members", len(members[c]))])
                 for c in classes if c in pf or len(members[c]) > 1]
    flags_new = [OrderedDict([("class", c), ("new_rel", c_animal[c]),
                              ("first_token", visible(new_first[c])),
                              ("prefix_of_other_labels", nf[c])])
                 for c in classes if c in nf]

    name = arm["relation"]
    info = catalog.get(name.lower()) or catalog.get(REL_ALIASES.get(name.lower(), "\0"))
    if info is None and rel in LOCAL_RELATIONS:
        o, d, r = LOCAL_RELATIONS[rel]
        info = {"source": o, "pre_rel_prompt_templates": None, "domain": d, "range": r}
    relation = OrderedDict([
        ("slug", rel), ("display_name", name), ("relation_type", arm["relation_type"]),
        ("subject_noun", arm["subject_noun"]), ("n_classes", len(classes)),
        ("pre_rel_source", info["source"] if info else None),
        ("pre_rel_prompt_templates", info["pre_rel_prompt_templates"] if info else None),
        ("pre_rel_domain", info["domain"] if info else None),
        ("pre_rel_range", info["range"] if info else None),
        ("newrel_prompt_notes", (arm.get("template_file") or {}).get("notes")),
    ])

    draw = OrderedDict([
        ("label_pool_path", arm["animal_pool"]["path"]),
        ("label_pool_sha256", prov["pool"]["sha256"]),
        ("label_pool_n", arm["animal_pool"]["n"]),
        ("label_pool_usable_for_this_tokenizer", pool["n_usable"]),
        ("target_format", pool["target_format"]),
        ("label_rule", f"{arm.get('label_rule')} / {pool['rule']}"),
        ("animal_subset_key", arm["seeds"]["animal_subset_key"]),
        ("bijection_key", arm["seeds"]["bijection_key"]),
        ("rng", arm["seeds"]["rng"]),
        ("rule", "subset = keyed_shuffle(usable pool, animal_subset_key)[:n_classes]; "
                 "classes in ASCENDING PRE-REL token id are zipped with "
                 "keyed_shuffle(subset, bijection_key)"),
    ])

    fh = {f: sha256_bytes(raw[f]) for f in ARM_FILES}
    provenance = OrderedDict([
        ("source", src.path), ("source_sha256", src.sha256),
        ("arm_dir_in_source", base), ("arm", arm["arm"]),
        ("file_sha256", fh),
        ("arm_generator", arm["generator"]), ("arm_generator_version", arm["generator_version"]),
        ("build_commit", arm["git_commit"]),
        ("tokenizer_of_manifest", arm["tokenizer_of_manifest"]),
        ("label_tokenizer", pool["tokenizer"]), ("label_tokenizer_revision", pool.get("revision")),
        ("decoded_with", f"{pool['tokenizer']}@{tok_rev}"),
        ("manifest", OrderedDict([("path", arm["manifest"]["path"]), ("sha256", arm["manifest"]["sha256"]),
                                  ("holdout_rule", arm["manifest"].get("holdout_rule"))])),
        ("records", OrderedDict([("path", arm["records"]["path"]), ("sha256", arm["records"]["sha256"]),
                                 ("n", arm["records"].get("n"))])),
        ("template_file", OrderedDict([("path", arm["template_file"]["path"]),
                                       ("sha256", arm["template_file"]["sha256"])])),
        ("arm_checks_passed", arm["checks"]["passed"]),
        ("n_train_subjects", n_t), ("n_heldout_subjects", n_h),
        ("n_support_subjects", len(arm["support_subjects"])),
        ("n_train_rows", len(rows["train"])), ("n_gate_rows", len(rows["gate"])),
        ("n_heldout_rows", len(rows["heldout"])),
    ])
    ex = arm.get("label_exclusion")
    render = OrderedDict([
        ("templates", prov["templates"]["path"]),
        ("template_file_path", arm["template_file"]["path"]),
        ("category_word", arm["template_file"]["category_word"]),
        ("label_pool_path", arm["animal_pool"]["path"]),
        ("pool_tokenizer", arm["animal_assertion"]["pool_tokenizer"]),
        ("asserted_tokenizers", arm["animal_assertion"]["asserted_tokenizers"]),
        ("role_seed", arm["seeds"]["role_seed"]),
        ("bijection_slug", arm["seeds"].get("bijection_slug")),
        ("relation_type", arm["relation_type"]),
        ("arm_name", arm["arm"]),
        ("generator_version", arm["generator_version"]),
        ("git_commit", arm["git_commit"]),
        ("label_exclusion", None if ex is None else OrderedDict([
            ("path", ex["path"]), ("sha256", ex["sha256"]), ("rule", ex["rule"]), ("k", ex["k"]),
            ("removed_from_usable", ex["removed_from_usable"]), ("draw_changed", ex["draw_changed"])])),
        ("build_provenance", prov),
    ])
    provenance["render"] = render
    cell = OrderedDict([("arm_set", arm_set_id), ("model", model), ("model_slug", model_slug),
                        ("relation", rel), ("repeat", repeat)])
    content = OrderedDict([("relation", relation), ("templates", tlist), ("draw", draw),
                           ("classes", cls_lines), ("excluded_classes", excl),
                           ("token_flags", OrderedDict([("pre_rel", flags_pre), ("new_rel", flags_new)])),
                           ("subjects", subj)])
    return cell, content, provenance

def aligned(rows: list[dict], indent: str) -> list[str]:
    if not rows:
        return []
    keys0 = list(rows[0].keys())
    common = [k for k in keys0 if all(k in r for r in rows)]

    ncol = 0
    for i, k in enumerate(common):
        if all(list(r.keys())[i] == k for r in rows):
            ncol = i + 1
        else:
            break
    frag = [[f"{jdump(k)}: {jdump(v)}" for k, v in r.items()] for r in rows]

    width = []
    for i in range(ncol):
        ls = sorted(len(f[i]) for f in frag)
        width.append(ls[min(len(ls) - 1, (95 * len(ls)) // 100)])
    out = []
    for r, f in zip(rows, frag):
        parts = []
        for i, p in enumerate(f):
            last = i == len(f) - 1
            s = p + ("" if last else ",")
            if i < ncol and not last:
                s = s.ljust(width[i] + 1)
            parts.append(s)
        out.append(indent + "{ " + " ".join(parts).rstrip() + " }")
    return out

def emit_list(out: list[str], key: str, items: list[str], comment: list[str],
              last: bool, indent="  "):
    for c in comment:
        out.append(f"{indent}// {c}".rstrip())
    if not items:
        out.append(f"{indent}{jdump(key)}: []" + ("" if last else ","))
        return
    out.append(f"{indent}{jdump(key)}: [")
    for i, l in enumerate(items):
        out.append(l + ("," if i < len(items) - 1 else ""))
    out.append(f"{indent}]" + ("" if last else ","))

def emit_obj(out, key, obj: dict, comment: list[str], last: bool, indent="  ", inline_keys=()):
    for c in comment:
        out.append(f"{indent}// {c}".rstrip())
    out.append(f"{indent}{jdump(key)}: {{")
    ks = list(obj.keys())
    w = max(len(jdump(k)) for k in ks)
    for i, k in enumerate(ks):
        v = obj[k]
        sep = "," if i < len(ks) - 1 else ""
        out.append(f"{indent}  {(jdump(k) + ':').ljust(w + 1)} {jdump(v)}{sep}")
    out.append(f"{indent}}}" + ("" if last else ","))

def header_block(cell, content, prov, manifest_info) -> list[str]:
    rel = content["relation"]
    L = [
        "=" * 100,
        f"NEW-REL, materialized: {rel['display_name']} ({cell['relation']}), repeat {cell['repeat']},",
        f"model {cell['model']}, arm set {cell['arm_set']}",
        "=" * 100,
        "",
        "WHAT THIS IS.  The fictional relation (NEW-REL) this one model was fine-tuned on in this one",
        "cell: the PRE-REL subjects are kept, each PRE-REL class is mapped by a keyed bijection to an",
        "animal label, and the prompts use the fictional templates below.  Everything the cell trained",
        "on is here: relation, templates, the class -> label map, every subject with its split.",
        "",
        "AS RUN.  This file reflects the arms AS RUN by the fleet build named in the arm set id.  A",
        "semantic-collision rule (or any pool pruning) adopted later produces a NEW arm set and a new",
        "sibling directory; this file is never rewritten to match it.",
        "",
        "HOW IT WAS GENERATED.  Arms: " + f"{prov['arm_generator']} v{prov['arm_generator_version']} at "
        f"linroute commit {prov['build_commit']}.",
        f"Arm read from {prov['source']}",
        f"  (sha256 {prov['source_sha256']}), member {prov['arm_dir_in_source']}/.",
        f"Materialized by {SCRIPT_PATH} v{VERSION} (linroute); deterministic, no model weights;",
        f"token ids decoded with the tokenizer only ({prov['decoded_with']}).",
        f"Label pool: {content['draw']['label_pool_path']}",
        f"  sha256 {content['draw']['label_pool_sha256']}, {content['draw']['label_pool_n']} labels.",
        f"Keys: animal subset {content['draw']['animal_subset_key']!s}; bijection {content['draw']['bijection_key']!s}.",
        "",
        "HOW TO READ IT.",
        "- \"class\" is the PRE-REL class id, which IS a token id: the scored (first non-degenerate)",
        "  token of \" \" + the PRE-REL answer under this model's tokenizer.  Answers that share that",
        "  token are one class (pre_rel_members lists them); pre_rel is the class's representative word.",
        "- new_rel is the NEW-REL label (whole word).  The loss and the restricted argmax only ever read",
        "  its FIRST token (new_rel_token_id); new_rel_n_tokens says how many tokens the whole word takes.",
        f"- Every *first_token field is tokenizer.decode([token_id]) with each space shown as {SPACE_MARK} (U+2423),",
        "  so a leading-space token reads like \"" + SPACE_MARK + "plant\".",
        "- Splits: train subjects are trained on with templates T1..T4 (4 rows each) and scored on T5",
        "  (the gate, 1 row each); heldout subjects are never trained on and are scored on all 5",
        "  templates (5 rows each): heldout is what NEW-REL-GEN measures.  support subjects are the five",
        "  PRE-REL few-shot demonstrations reserved by admission; they appear in no NEW-REL row, so their",
        "  new_rel is null.",
        "- pre_rel_object appears on a subject line only when its own PRE-REL answer differs from the",
        "  class representative (same first token, different word).",
        "- token_flags lists classes whose first token is a prefix of another class's whole word, or",
        "  that merge several PRE-REL answers (n_merged_members > 1; the class line lists them); and",
        "  NEW-REL labels whose first token begins another label of this cell.",
        "",
        "FORMAT.  JSON5 with quoted keys and no trailing commas.  Every comment is a whole line whose",
        "first non-blank characters are //; deleting those lines leaves strict JSON.",
        "=" * 100,
    ]
    return ["// " + l if l else "//" for l in L]

def render_cell(cell, content, prov, manifest_info) -> str:
    out = header_block(cell, content, prov, manifest_info)
    out.append("{")
    emit_obj(out, "cell", cell, ["Which cell this is: one fine-tuning run = (model, relation, repeat)."], False)
    emit_obj(out, "relation", content["relation"],
             ["The relation.  pre_rel_* describe the SOURCE (PRE-REL) relation; newrel_prompt_notes is the",
              "template author's note on how the fictional prompts avoid naming it."], False)
    emit_list(out, "templates", aligned(content["templates"], "    "),
              ["The NEW-REL prompt templates ({} = subject).  T1..T4 train, T5 is the gate/evaluation",
               "template.  The example is the rendered row for the first training subject."], False)
    emit_obj(out, "draw", content["draw"],
             ["How the labels were drawn: pool, keys and the bijection rule."], False)
    emit_list(out, "classes", aligned(content["classes"], "    "),
              ["The class -> label map, one line per retained PRE-REL class, in ascending token id (the",
               "order the bijection zips).  n_train / n_heldout count subjects."], False)
    emit_list(out, "excluded_classes", aligned(content["excluded_classes"], "    "),
              ["PRE-REL classes the admission step did NOT retain for this model (no NEW-REL label)."], False)
    out.append("  // Token-level flags (see the header).  Empty lists mean nothing to flag.")
    out.append('  "token_flags": {')
    tf = content["token_flags"]
    emit_list(out, "pre_rel", aligned(tf["pre_rel"], "      "), [], False, indent="    ")
    emit_list(out, "new_rel", aligned(tf["new_rel"], "      "), [], True, indent="    ")
    out.append("  },")
    emit_list(out, "subjects", aligned(content["subjects"], "    "),
              ["Every subject, one per line: train and heldout in the arm's (manifest) order, then support."],
              False)
    prov = OrderedDict(prov)
    render = prov.pop("render", None)
    emit_obj(out, "provenance", prov,
             ["Where this cell's arm came from, with the sha256 of every arm file as packed."],
             render is None)
    if render is not None:
        emit_obj(out, "render", render,
                 ["What scripts/arms_from_materialized.py needs, besides the definition above (templates,",
                  "the class -> label map, the subjects), to re-render this cell's packed arm byte for byte:",
                  "the builder's arguments, the build stamps, the label exclusion (null if none) and the",
                  "fleet's build-provenance record."], True)
    out.append("}")
    return "\n".join(out) + "\n"

def summarize(rows_for_rr):
    def groups(key):
        g = defaultdict(list)
        for r in rows_for_rr:
            g[r[key]].append(r["model"])
        return [sorted(v) for _, v in sorted(g.items(), key=lambda kv: (-len(kv[1]), sorted(kv[1])))]
    bij = groups("bijection_sha256")
    return OrderedDict([
        ("n_models", len(rows_for_rr)),
        ("bijection_identical_across_models", (len(bij) == 1) if len(rows_for_rr) > 1 else None),
        ("bijection_groups", bij),
        ("subject_content_groups", groups("subjects_sha256")),
        ("subject_split_groups", groups("splits_sha256")),
        ("class_set_groups", groups("class_words_sha256")),
        ("n_classes_by_model", OrderedDict(sorted((r["model"], r["n_classes"]) for r in rows_for_rr))),
    ])

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", action="append",
                    help="a packed inputs tar (or a directory) holding arms/<model>/<rel>/r<k>/; repeatable "
                         "(required unless --html-only)")
    ap.add_argument("--arm-set-id", required=True)
    ap.add_argument("--out", required=True, help="parent directory; output goes to <out>/<arm-set-id>/")
    ap.add_argument("--html", help="also write the viewer pages into this directory")
    ap.add_argument("--html-json5-prefix", default="data",
                    help="where, relative to the HTML directory, the JSON5 tree is served")
    ap.add_argument("--html-only", action="store_true",
                    help="write only the viewer pages (--html), from the JSON5 files already under "
                         "<out>/<arm-set-id>/; loads no tokenizer and no tar, and writes nothing under <out>")
    ap.add_argument("--arm-set-label", default=None,
                    help="short human name of --arm-set-id in the viewer (e.g. 'as run'); default: the id")
    ap.add_argument("--also-arm-set", action="append", default=[], metavar="ID:SUBDIR:LABEL",
                    help="--html-only: also render the already-materialized arm set ID into <html>/SUBDIR, "
                         "link the two with an arm-set switch, and show on every relation page (and in both "
                         "indexes) which cells it changed relative to --arm-set-id; repeatable")
    ap.add_argument("--copy-json5", action="store_true",
                    help="--html-only: also copy each rendered arm set's JSON5 tree to "
                         "<html>/<html-json5-prefix>/<id>/, so the pages' download links resolve")
    ap.add_argument("--inventory-root", default=str(REPO_ROOT / "data"),
                    help="where relations/, relations-bear10/ and relations/engineered/ inventory summaries "
                         "live, for the viewer's subject funnel (default: this repo's data/)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate", action="store_true", help="parse every written file back")
    a = ap.parse_args(argv)

    need(re.fullmatch(r"[A-Za-z0-9._-]+", a.arm_set_id), "arm set id: letters, digits, . _ - only")
    root = Path(a.out) / a.arm_set_id
    if a.html_only:
        need(a.html, "--html-only needs --html")
        html_only(a)
        return 0
    need(not a.also_arm_set and not a.copy_json5, "--also-arm-set and --copy-json5 need --html-only")
    need(a.arms, "--arms is required (unless --html-only)")
    if root.exists() and any(root.iterdir()) and not a.overwrite:
        raise Fail(f"{root} is not empty; pass --overwrite to regenerate in place")

    sources = [Source(p) for p in a.arms]
    toks = Tokenizers()
    rec_cache = {}
    cells = []
    seen = set()
    for src in sources:
        catalog = relation_catalog(src.files)
        for m, rel, rd in src.cells():
            need((m, rel, rd) not in seen, f"cell {m}/{rel}/{rd} appears in two sources")
            seen.add((m, rel, rd))
            cell, content, prov = build_cell(m, rel, rd, src, toks, rec_cache, catalog, a.arm_set_id)
            cells.append((cell, content, prov, src))
            print(f"CELL {m}/{rel}/{rd} classes={len(content['classes'])}", flush=True)

    commits = sorted({p["build_commit"] for _, _, p, _ in cells})
    pools = sorted({(c["draw"]["label_pool_path"], c["draw"]["label_pool_sha256"], c["draw"]["label_pool_n"])
                    for _, c, _, _ in cells})
    manifest_info = {"commits": commits, "pools": pools}

    index_rows = []
    written = {}
    for cell, content, prov, src in cells:
        rel_path = f"{cell['model_slug']}/{cell['relation']}/r{cell['repeat']}.json5"
        text = render_cell(cell, content, prov, manifest_info)
        written[rel_path] = text.encode("utf-8")
        cm = OrderedDict((c["pre_rel"], c["new_rel"]) for c in content["classes"])
        index_rows.append(OrderedDict([
            ("model", cell["model"]), ("relation", cell["relation"]), ("repeat", cell["repeat"]),
            ("path", rel_path),
            ("n_classes", len(content["classes"])),
            ("n_train_subjects", prov["n_train_subjects"]), ("n_heldout_subjects", prov["n_heldout_subjects"]),
            ("n_support_subjects", prov["n_support_subjects"]),
            ("n_train_rows", prov["n_train_rows"]), ("n_gate_rows", prov["n_gate_rows"]),
            ("n_heldout_rows", prov["n_heldout_rows"]),
            ("n_flags_pre_rel", len(content["token_flags"]["pre_rel"])),
            ("n_flags_new_rel", len(content["token_flags"]["new_rel"])),
            ("file_sha256", sha256_bytes(written[rel_path])),
            ("content_sha256", sha256_bytes(canon(content))),
            ("subjects_sha256", sha256_bytes(canon(content["subjects"]))),
            ("splits_sha256", sha256_bytes(canon(sorted([s["subject"], s["split"]] for s in content["subjects"])))),
            ("bijection_sha256", sha256_bytes(canon(sorted(cm.items())))),
            ("class_words_sha256", sha256_bytes(canon(sorted(cm)))),
            ("class_map", cm),
        ]))
    index_rows.sort(key=lambda r: (r["relation"], r["repeat"], r["model"]))
    by_rr = defaultdict(list)
    for r in index_rows:
        by_rr[(r["relation"], r["repeat"])].append(r)
    summaries = [OrderedDict([("relation", k[0]), ("repeat", k[1])] + list(summarize(v).items()))
                 for k, v in sorted(by_rr.items())]

    I = ["// " + l if l else "//" for l in [
        f"Index of arm set {a.arm_set_id}: one row per (model, relation, repeat), then one summary per",
        "(relation, repeat).  Written by " + f"{SCRIPT_PATH} v{VERSION}; see manifest.json5 and ../README.md.",
        "",
        "Row hashes (sha256), for diffing two arm sets row by row:",
        "- file_sha256: the .json5 file's bytes (includes the arm set id and source paths).",
        "- content_sha256: the cell's content only (relation, templates, draw keys, classes, excluded",
        "  classes, token flags, subjects), canonical JSON; equal across arm sets iff the cell is unchanged.",
        "- subjects_sha256: the subject lines (subject, class, pre_rel, new_rel, split).",
        "- splits_sha256: subject strings and splits only, ignoring every label.",
        "- bijection_sha256: the PRE-REL word -> NEW-REL label map.  class_words_sha256: its PRE-REL words.",
        "class_map is PRE-REL representative word -> NEW-REL label, in ascending PRE-REL token id.",
        "",
        "Summaries: *_groups list groups of models whose hash is identical, largest group first.",
        "",
        "Every comment is a whole // line; deleting them leaves strict JSON.",
    ]]
    I.append("{")
    I.append(f'  "arm_set": {jdump(a.arm_set_id)},')
    I.append("  // One row per cell.")
    I.append('  "cells": [')
    for i, r in enumerate(index_rows):
        I.append("    {")
        ks = list(r.keys())
        w = max(len(jdump(k)) for k in ks)
        for j, k in enumerate(ks):
            I.append(f"      {(jdump(k) + ':').ljust(w + 1)} {jdump(r[k])}" + ("," if j < len(ks) - 1 else ""))
        I.append("    }" + ("," if i < len(index_rows) - 1 else ""))
    I.append("  ],")
    I.append("  // What is shared across models, per (relation, repeat).")
    I.append('  "summaries": [')
    for i, s in enumerate(summaries):
        I.append("    {")
        ks = list(s.keys())
        w = max(len(jdump(k)) for k in ks)
        for j, k in enumerate(ks):
            I.append(f"      {(jdump(k) + ':').ljust(w + 1)} {jdump(s[k])}" + ("," if j < len(ks) - 1 else ""))
        I.append("    }" + ("," if i < len(summaries) - 1 else ""))
    I.append("  ]")
    I.append("}")
    written["index.json5"] = ("\n".join(I) + "\n").encode("utf-8")

    models = sorted({c["model"] for c, _, _, _ in cells})
    tokrevs = OrderedDict(sorted({(p["label_tokenizer"], p["label_tokenizer_revision"]) for _, _, p, _ in cells}))
    mf = OrderedDict([
        ("arm_set", a.arm_set_id),
        ("description", "The NEW-REL arms AS RUN by this fleet build.  A later collision rule or pool "
                        "pruning produces a new arm set and a sibling directory; nothing here is rewritten."),
        ("build_commits", commits),
        ("arm_generator", sorted({f"{p['arm_generator']} v{p['arm_generator_version']}" for _, _, p, _ in cells})),
        ("label_pools", [OrderedDict([("path", p), ("sha256", s), ("n", n)]) for p, s, n in pools]),
        ("sources", [OrderedDict([
            ("path", s.path), ("kind", s.kind), ("sha256", s.sha256), ("md5", s.md5), ("bytes", s.size),
            ("n_cells", sum(1 for *_, src in cells if src is s)),
            ("models", sorted({c["model"] for c, _, _, src in cells if src is s}))]) for s in sources]),
        ("materializer", OrderedDict([("script", SCRIPT_PATH), ("version", VERSION),
                                      ("script_sha256", sha256_file(Path(__file__)))])),
        ("tokenizers", tokrevs),
        ("n_cells", len(cells)), ("n_models", len(models)),
        ("n_relations", len({c["relation"] for c, *_ in cells})),
        ("n_relation_repeats", len(by_rr)),
        ("n_relation_repeats_bijection_identical_across_models",
         sum(1 for s in summaries if s["bijection_identical_across_models"] is True)),
        ("n_relation_repeats_with_several_models",
         sum(1 for s in summaries if s["n_models"] > 1)),
        ("models", models),
    ])
    M = ["// " + l if l else "//" for l in [
        f"Arm set {a.arm_set_id}: identity of the arms materialized in this directory.",
        "sources are the packed inputs tars the fleet(s) trained on, read without extraction; their",
        "sha256/md5 are what the fleet checked before launch.  See ../README.md for the layout.",
        "Every comment is a whole // line; deleting them leaves strict JSON.",
    ]]
    M.append("{")
    ks = list(mf.keys())
    w = max(len(jdump(k)) for k in ks)
    for j, k in enumerate(ks):
        M.append(f"  {(jdump(k) + ':').ljust(w + 1)} {jdump(mf[k])}" + ("," if j < len(ks) - 1 else ""))
    M.append("}")
    written["manifest.json5"] = ("\n".join(M) + "\n").encode("utf-8")

    for rel_path, b in sorted(written.items()):
        p = root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    if a.validate:
        for rel_path in sorted(written):
            t = (root / rel_path).read_text("utf-8")
            json.loads(strip_comment_lines(t))
        print(f"VALIDATE strict-json-after-comment-strip ok={len(written)}")
    print(f"WROTE {len(written)} files under {root} bytes={sum(len(b) for b in written.values())}")

    if a.html:
        write_html(Path(a.html), a.arm_set_id, a.html_json5_prefix, cells, index_rows, summaries, mf,
                   Path(a.inventory_root), label=a.arm_set_label or a.arm_set_id)
    return 0

def html_only(a):
    sets = [(a.arm_set_id, ".", a.arm_set_label or a.arm_set_id)]
    for spec in a.also_arm_set:
        parts = spec.split(":", 2)
        need(len(parts) == 3 and all(parts), f"--also-arm-set {spec!r}: want ID:SUBDIR:LABEL")
        need(re.fullmatch(r"[A-Za-z0-9._-]+", parts[0]) and re.fullmatch(r"[A-Za-z0-9._-]+", parts[1])
             and parts[1] not in (".", ".."), f"--also-arm-set {spec!r}: bad id or subdir")
        sets.append(tuple(parts))
    loaded = {sid: load_materialized(Path(a.out) / sid) for sid, _, _ in sets}
    base_id, _, base_label = sets[0]
    diffs = [(base_label, lab, arm_set_diff(loaded[base_id], loaded[sid])) for sid, _, lab in sets[1:]]
    html_root = Path(a.html)
    for sid, sub, lab in sets:
        outdir = html_root if sub == "." else html_root / sub
        up = "" if sub == "." else "../"
        nav = [(l2, i2, os.path.relpath(html_root if s2 == "." else html_root / s2, outdir).replace(os.sep, "/"))
               for i2, s2, l2 in sets]
        cells, index_rows, summaries, mf = loaded[sid]
        write_html(outdir, sid, up + a.html_json5_prefix, cells, index_rows, summaries, mf,
                   Path(a.inventory_root), label=lab, up=up, nav=nav, diffs=diffs)
        if a.copy_json5:
            src_root, dst_root = Path(a.out) / sid, html_root / a.html_json5_prefix / sid
            n = 0
            for f in sorted(src_root.rglob("*.json5")):
                d = dst_root / f.relative_to(src_root)
                b = f.read_bytes()
                if not d.is_file() or d.read_bytes() != b:
                    d.parent.mkdir(parents=True, exist_ok=True)
                    d.write_bytes(b)
                    n += 1
            print(f"JSON5 {sid}: {n} files copied or updated under {dst_root}")

def arm_set_diff(base, extra) -> dict:
    b_cells = {(c["model"], c["relation"], c["repeat"]): (c, ct, pv, rd) for c, ct, pv, rd in base[0]}
    e_cells = {(c["model"], c["relation"], c["repeat"]): (c, ct, pv, rd) for c, ct, pv, rd in extra[0]}
    b_rows = {(r["model"], r["relation"], r["repeat"]): r for r in base[1]}
    e_rows = {(r["model"], r["relation"], r["repeat"]): r for r in extra[1]}
    out = defaultdict(lambda: {"common": [], "changed": [], "only_base": [], "only_extra": [],
                               "removed": {}, "splits_same": True, "class_words_same": True})
    for k in sorted(set(b_rows) | set(e_rows)):
        info = out[k[1]]
        if k not in e_rows:
            info["only_base"].append(k)
            continue
        if k not in b_rows:
            info["only_extra"].append(k)
            continue
        info["common"].append(k)
        ex = ((e_cells[k][3] or {}).get("label_exclusion") or {})
        info["removed"][k] = (tuple(ex.get("removed_from_usable") or ()), ex.get("rule"), ex.get("k"),
                              ex.get("draw_changed"))
        if b_rows[k]["content_sha256"] != e_rows[k]["content_sha256"]:
            info["changed"].append((k, b_cells[k][1], e_cells[k][1]))
            info["splits_same"] &= b_rows[k]["splits_sha256"] == e_rows[k]["splits_sha256"]
            info["class_words_same"] &= b_rows[k]["class_words_sha256"] == e_rows[k]["class_words_sha256"]
    return dict(out)

def load_materialized(root: Path):
    def load(p: Path):
        need(p.is_file(), f"--html-only: {p} missing")
        return json.loads(strip_comment_lines(p.read_text("utf-8")), object_pairs_hook=OrderedDict)
    idx = load(root / "index.json5")
    mf = load(root / "manifest.json5")
    cells = []
    for r in idx["cells"]:
        d = load(root / r["path"])
        need(d["cell"]["model"] == r["model"] and d["cell"]["relation"] == r["relation"]
             and d["cell"]["repeat"] == r["repeat"], f"--html-only: {r['path']} disagrees with index.json5")
        content = OrderedDict((k, d[k]) for k in ("relation", "templates", "draw", "classes",
                                                  "excluded_classes", "token_flags", "subjects"))

        cells.append((d["cell"], content, d["provenance"], d.get("render")))
    return cells, idx["cells"], idx["summaries"], mf

def esc(s) -> str:
    return html.escape("" if s is None else str(s))

_SET_NAMES = None

def set_name(key: str) -> str:
    global _SET_NAMES
    if _SET_NAMES is None:
        reg = (Path(os.environ.get("LIN_PAPER_REPO") or Path(__file__).resolve().parents[2] / "paper")
               / "metadata" / "relation-sets.json5")
        if reg.is_file():
            sd = json.loads(reg.read_text("utf-8"))["source_display"]
            _SET_NAMES = {k: v["name"] for k, v in sd.items() if not k.startswith("_")}
        else:
            print(f"WARNING: {reg} not found; relation-set names shown as their internal keys", file=sys.stderr)
            _SET_NAMES = {}
    return _SET_NAMES.get(key, key)

_TERMS = None

def term_name(key: str, technical: str) -> str:
    global _TERMS
    if _TERMS is None:
        shared = (Path(os.environ.get("LIN_PAPER_REPO") or Path(__file__).resolve().parents[2] / "paper")
                  / "plots" / "shared")
        try:
            sys.path.insert(0, str(shared))
            import terms
            _TERMS = terms.load()
        except Exception as err:
            print(f"WARNING: paper term registry unavailable ({err}); technical measure names shown",
                  file=sys.stderr)
            _TERMS = False
    return _TERMS.term(key) if _TERMS else technical

def shown_rel_name(name: str) -> str:
    return re.sub(r"^engineered (?=town\b)", "", name or "")

def shown_pre_rel_source(text: str) -> str:
    return (text or "").replace("linroute engineered relation", f"linroute {set_name('engineered')} relation")

def tw(word, first_token, skipped=None, token_id=None) -> str:
    w = "" if word is None else str(word)
    if not first_token:
        return esc(w)
    core = first_token.replace(SPACE_MARK, " ").lstrip(" ")
    sk = (skipped or "").replace(SPACE_MARK, " ").lstrip(" ")
    pre = sk + core
    if core and w[:len(pre)].lower() == pre.lower():
        return f"{esc(w[:len(sk)])}<u>{esc(w[len(sk):len(pre)])}</u>{esc(w[len(pre):])}"
    if core and set(core) == {"\ufffd"} and w[len(sk):]:
        i = len(sk)
        note = ("first token is a partial UTF-8 byte of this letter"
                + (f" (token id {token_id})" if token_id is not None else ""))
        return (f'{esc(w[:i])}<u class="pb" title="{esc(note)}">{esc(w[i])}</u>{esc(w[i + 1:])}'
                f'<sup class="pbn" title="{esc(note)}">partial byte'
                + (f" {token_id}" if token_id is not None else "") + "</sup>")
    return f'{esc(w)}<span class="ftb">{esc(first_token)}</span>'

def pre_html(c) -> str:
    return (f'<span class="tw pre">{tw(c["pre_rel"], c["pre_rel_first_token"], c.get("pre_rel_skipped_prefix"), c["class"])}'
            "</span>")

def new_html(c) -> str:
    return f'<span class="tw new">{tw(c["new_rel"], c["new_rel_first_token"], None, c["new_rel_token_id"])}</span>'

TOKEN_LEGEND = ('<p class="tok-legend">Reading the words: <span class="tw"><u>crit</u>ter</span> is the whole '
                'word with its <u class="lg">first token</u> marked; the loss and the restricted argmax read only '
                'that token (tokens decoded with each model&rsquo;s own tokenizer). A dotted first letter, as in '
                '<span class="tw"><u class="pb">&#352;</u>koda<sup class="pbn">partial byte</sup></span>, means a '
                'byte-level tokenizer split that non-ASCII letter, so the first token is only part of its UTF-8 '
                'bytes; the note gives its token id.</p>')

def chips(words, cls) -> str:
    return "".join(f'<span class="chip {cls}">{esc(w)}</span>' for w in words)

def card_body(heldout, train, support) -> str:
    h = (f'<div class="hbox"><span class="lab">held-out ({len(heldout)})</span>{chips(heldout, "h")}</div>'
         if heldout else '<div class="hbox none"><span class="lab">held-out</span>none</div>')
    if train:
        h += (f'<details class="tbox"><summary>train ({len(train)})</summary>'
              f'{esc(", ".join(train))}</details>')
    if support:
        h += (f'<div class="sbox"><span class="lab">support: few-shot only, in no NEW-REL row</span>'
              f'{esc(", ".join(support))}</div>')
    return h

def inventory_for(inv_root: Path, rel: str):
    for sub in ("relations", "relations-bear10"):
        p = inv_root / sub / rel / "inventory" / "summary.json"
        if p.is_file():
            return "inventory", json.loads(p.read_text("utf-8")), p
    if rel.startswith("engineered_town_"):
        name = rel[len("engineered_town_"):]
        name = (name[:-len("_band")] if name.endswith("_band") else name).replace("_", "-")
        p = inv_root / "relations" / "engineered" / name / "summary.json"
        if p.is_file():
            return "engineered", json.loads(p.read_text("utf-8")), p
    return None

REASON_TEXT = {
    "unresolved_object_conflict": "rows whose subject string has more than one distinct answer (unresolved "
                                  "object conflicts; all such rows go)",
    "empty_object": "rows with an empty answer",
    "all_tokens_degenerate": "rows whose answer has no non-degenerate token",
}

def funnel_section(rel, models, lowest, inv, inv_root: Path, short) -> list[str]:
    P = ['<h2 id="funnel">Where the subjects went</h2>',
         '<p class="note">Admission rules (linroute <code>docs/admission.md</code>): five subjects are first '
         'reserved as PRE-REL few-shot demonstrations, then a class keeps its NEW-REL label only if it still has '
         'at least 5 subjects, and a relation needs at least 5 such classes; each kept class then gives up its '
         'held-out subjects by the <code>mushroom-v2</code> rule. Classes are token ids, so all of this runs per '
         'tokenizer. Models with identical counts are grouped.</p>']
    kind = inv[0] if inv else None
    summ = inv[1] if inv else None
    src_rows = src_dup = None
    if kind == "inventory":
        pts = summ.get("per_tokenizer") or []
        srcs = {p["n_source_rows"] for p in pts}
        dups = {p["n_exact_duplicate_rows"] for p in pts}
        if len(srcs) == 1 and len(dups) == 1:
            src_rows, src_dup = srcs.pop(), dups.pop()
        src = summ.get("source") or {}
        where = src.get("source_file") or src.get("relation_file") or ""
        line = (f'<p class="note">Source: <b>{src_rows}</b> rows in <code>{esc(where)}</code>'
                if src_rows is not None else f'<p class="note">Source: <code>{esc(where)}</code>')
        bc = ((src.get("declared") or {}).get("build_counts")) or {}
        if bc:
            drops = [f"{v} {k[len('n_'):].replace('_', ' ')}" for k, v in sorted(bc.items())
                     if v and k not in ("n_input_pairs", "n_samples", "n_unique_objects")]
            line += (f' (the pack build kept {bc.get("n_samples")} of {bc.get("n_input_pairs")} input pairs'
                     + (": dropped " + ", ".join(drops) if drops else "") + ")")
        rp = inv[2].relative_to(inv_root.parent) if inv[2].is_relative_to(inv_root.parent) else inv[2]
        have = sorted(p["tokenizer"] for p in pts)
        miss = [m for m in models if m not in have]
        line += (f'. Counts per tokenizer: <code>{esc(rp)}</code>'
                 + (f', which covers {len(have)} of these {len(models)} models; for the other {len(miss)} the '
                    'per-reason breakdown before admission is not committed, and the numbers come from the arm '
                    'files alone.' if miss else '.') + '</p>')
        P.append(line)
    elif kind == "engineered":
        rp = inv[2].relative_to(inv_root.parent) if inv[2].is_relative_to(inv_root.parent) else inv[2]
        st = summ.get("attribute_status_counts") or {}
        cs = summ.get("class_support") or {}
        P.append(f'<p class="note">Source: {summ.get("n_rows")} towns (<code>{esc(rp)}</code>); attribute status: '
                 + ", ".join(f"{esc(k.replace('_', ' '))} {v}" for k, v in sorted(st.items()))
                 + (f'; {cs["unknown"]} towns fall in no class (&ldquo;unknown&rdquo;)' if "unknown" in cs else "")
                 + f'. No per-tokenizer inventory is committed for the {esc(set_name("engineered"))} relations, so the per-model numbers '
                   'come from the arm files alone.</p>')
    else:
        P.append('<p class="note">No committed inventory summary was found for this relation; the numbers below '
                 'come from the arm files alone.</p>')

    groups = OrderedDict()
    for m in models:
        ct, prov = lowest[m]
        pts = {p["tokenizer"]: p for p in (summ.get("per_tokenizer") or [])} if kind == "inventory" else {}
        p = pts.get(m)
        T, H, S = prov["n_train_subjects"], prov["n_heldout_subjects"], prov["n_support_subjects"]
        excl = ct["excluded_classes"]
        ex_sum = sum(int(x["support"] or 0) for x in excl)
        pool = T + H + S + ex_sum
        rec_n = (prov.get("records") or {}).get("n")
        L = []
        marked = None
        if p:
            L.append(f'<b>{p["n_source_rows"]}</b> source rows')
            L.append(f'<b>{p["n_records"]}</b> after removing '
                     + (f'{p["n_exact_duplicate_rows"]} exact duplicate rows' if p["n_exact_duplicate_rows"]
                        else "exact duplicates (there were none)"))
            reasons = [(k, v) for k, v in p["excluded_by_reason"].items() if v]
            if reasons:
                parts = []
                for k, v in reasons:
                    t = f"{v} {REASON_TEXT.get(k, k.replace('_', ' '))}"
                    if k == "unresolved_object_conflict" and p.get("n_conflicting_subject_strings"):
                        t = (f'{v} rows with unresolved object conflicts: {p["n_conflicting_subject_strings"]} '
                             'subject strings each had more than one distinct answer, and all their rows go')
                    parts.append(t)
                L.append(f'<b>{p["n_included"]}</b> after dropping ' + "; ".join(parts))
            else:
                L.append(f'<b>{p["n_included"]}</b> included (no rows dropped for conflicts or empty answers)')
            if p.get("n_context_mismatched"):
                L.append(f'<b>{p["n_context_consistent"]}</b> after admission drops {p["n_context_mismatched"]} '
                         'records whose answer token differs in the rendered prompt')
            if p["n_context_consistent"] != pool:
                L.append(f'<span class="warn-t">the arm accounts for {pool} eligible subjects, not '
                         f'{p["n_context_consistent"]}; {p["n_context_consistent"] - pool} are unaccounted for</span>')
        else:
            if src_rows is not None:
                L.append(f'<b>{src_rows}</b> source rows (the source file&rsquo;s count from the committed '
                         'inventory; it does not depend on the tokenizer)')
            if rec_n is not None:
                t = f'<b>{rec_n}</b> records in the records file this arm read'
                if src_rows is not None and src_rows - src_dup == rec_n:
                    t += (f', that is after removing {src_dup} exact duplicate rows' if src_dup
                          else ' (no exact duplicates)')
                L.append(t)
                why = ("no per-tokenizer breakdown is committed; compare the towns in no class above"
                       if kind == "engineered" else
                       "conflicting or unusable answers; the per-reason breakdown for this tokenizer is not "
                       "committed")
                L.append(f'<b>{pool}</b> eligible at admission'
                         + (f'; {rec_n - pool} records were dropped before admission ({why})'
                            if rec_n != pool else ''))
            else:
                L.append(f'<b>{pool}</b> eligible at admission')
        L.append(f'<b>{pool - S}</b> after reserving {S} few-shot support subjects')
        if excl:
            by_reason = OrderedDict()
            for x in sorted(excl, key=lambda x: (-(x["support"] or 0), str(x["pre_rel"]))):
                by_reason.setdefault(x["reason"], []).append(x)
            n_all = len(ct["classes"]) + len(excl)
            for reason, xs in by_reason.items():
                what = ("support below 5" if reason == "support_below_min" else esc(reason))
                s_sum = sum(int(x["support"] or 0) for x in xs)
                head = (f'<b>{pool - S - ex_sum}</b> after dropping {len(xs)} of {n_all} classes for {what}, '
                        f'which held {s_sum} subjects' if len(by_reason) == 1 else
                        f'{len(xs)} of {n_all} classes dropped for {what}, holding {s_sum} subjects')

                def listing(words):
                    lst = ", ".join(f'{w}&nbsp;({x["support"]})' for w, x in zip(words, xs))
                    return head + (f': {lst}' if len(xs) <= 12 else
                                   f' <details class="exl"><summary>list the {len(xs)} classes '
                                   f'(support in parentheses)</summary>{lst}</details>')
                L.append(listing([esc(x["pre_rel"]) for x in xs]))
                marked = marked or {}
                marked[len(L) - 1] = listing([f'<span class="tw">{tw(x["pre_rel"], x["pre_rel_first_token"], None, x["class"])}'
                                              '</span>' for x in xs])
            if len(by_reason) > 1:
                L.append(f'<b>{pool - S - ex_sum}</b> in the kept classes')
        L.append(f'Final: <b>{T}</b> train + <b>{H}</b> held-out subjects in {len(ct["classes"])} classes')

        key = (bool(p), tuple(L))
        groups.setdefault(key, []).append((m, tuple(sorted((marked or {}).items()))))
    for (has_inv, L), members in groups.items():
        ms = [m for m, _ in members]
        L = list(L)
        marks = {mk for _, mk in members}
        tail = ""
        if len(marks) == 1:
            for i, line in marks.pop():
                L[i] = line
        else:
            tail = ('<p class="muted fg-t">The dropped classes&rsquo; first tokens differ across these models; '
                    'the one-cell view below shows each model&rsquo;s.</p>')
        P.append(f'<div class="fg"><div class="fg-m">{esc(", ".join(short(m) for m in ms))}'
                 + ('' if has_inv or kind != "inventory" else ' <span class="muted">(arm files only)</span>')
                 + '</div><ol>' + "".join(f"<li>{l}</li>" for l in L) + "</ol>" + tail + "</div>")
    return P

def maps_section(rel, sums, content_of, short) -> list[str]:
    P = ['<h2 id="maps">Class maps and held-out subjects</h2>', TOKEN_LEGEND,
         '<p class="note">One block per map group (models with the same PRE-REL word &rarr; NEW-REL label map), '
         'split further when the same words have different first tokens under different tokenizers. In each '
         f'class card the <b>held-out</b> subjects (never trained on; what {esc(term_name("generalization", "NEW-REL-GEN"))} '
         'is measured on) are boxed, the '
         'train subjects are folded away, and the support subjects (PRE-REL few-shot demonstrations) are listed '
         'apart because they appear in no NEW-REL row. Splits are per tokenizer: when models of one block '
         'differ, each variant is listed under the models it belongs to.</p>']
    blocks = []
    for s in sums:
        rr = s["repeat"]
        for gi, g in enumerate(s["bijection_groups"], 1):
            sub = OrderedDict()
            for m in g:
                ct = content_of[(m, rel, rr)][1]
                sig = tuple(sorted((c["pre_rel"], c["pre_rel_first_token"], c.get("pre_rel_skipped_prefix") or "",
                                    c["new_rel"], c["new_rel_first_token"]) for c in ct["classes"]))
                sub.setdefault(sig, []).append(m)
            subs = sorted(sub.values(), key=lambda ms: -len(ms))
            for vi, ms in enumerate(subs):
                blocks.append((rr, gi, vi, len(subs), ms))
    n_open = len(blocks) if len(blocks) <= 4 else 1
    for bi, (rr, gi, vi, nv, ms) in enumerate(blocks):
        cts = {m: content_of[(m, rel, rr)][1] for m in ms}
        ct0 = cts[ms[0]]
        label = f"r{rr} &middot; map group {gi}" + (f" &middot; first-token variant {chr(97 + vi)} of {nv}"
                                                     if nv > 1 else "")
        H = [f'<details class="mg"{" open" if bi < n_open else ""}><summary><b>{label}</b> &middot; '
             f'{len(ms)} model{"s" if len(ms) != 1 else ""}: {esc(", ".join(short(m) for m in ms))}</summary>',
             '<div class="cards">']
        for c0 in ct0["classes"]:
            variants = OrderedDict()
            for m in ms:
                ct = cts[m]
                cid = next(c["class"] for c in ct["classes"] if c["pre_rel"] == c0["pre_rel"])
                b = {"train": [], "heldout": [], "support": []}
                for x in ct["subjects"]:
                    if x["class"] == cid:
                        b[x["split"]].append(x["subject"])
                variants.setdefault((tuple(b["heldout"]), tuple(b["train"]), tuple(b["support"])), []).append(m)
            body = []
            for (h, t, sp), vms in variants.items():
                if len(variants) > 1:
                    body.append(f'<div class="vm">{esc(", ".join(short(m) for m in vms))}</div>')
                body.append(card_body(list(h), list(t), list(sp)))
            H.append(f'<div class="ccard"><div class="cmap">{pre_html(c0)} <span class="arr">&rarr;</span> '
                     f'{new_html(c0)}</div>{"".join(body)}</div>')
        H.append("</div>")

        orphan = OrderedDict()
        for m in ms:
            ct = cts[m]
            kept = {c["class"] for c in ct["classes"]}
            o = tuple(f'{x["subject"]} ({x["pre_rel"]})' for x in ct["subjects"]
                      if x["split"] == "support" and x["class"] not in kept)
            if o:
                orphan.setdefault(o, []).append(m)
        for o, oms in orphan.items():
            H.append(f'<p class="note sbox">Support subjects of excluded classes (no NEW-REL label)'
                     + (f' for {esc(", ".join(short(m) for m in oms))}' if len(orphan) > 1 or len(oms) < len(ms)
                        else "") + f': {esc(", ".join(o))}</p>')
        H.append("</details>")
        P.append("".join(H))
    return P

def nav_html(nav, arm_set, target) -> str:
    if not nav or len(nav) < 2:
        return ""
    parts = []
    for lab, sid, href in nav:
        txt = f'{esc(lab)} <code>{esc(sid)}</code>'
        if sid == arm_set:
            parts.append(f'<b>{txt}</b>')
        else:
            parts.append(f'<a href="{esc(href.rstrip("/") + "/" + target)}">{txt}</a>')
    return '<p class="armset">Arm set: ' + " &middot; ".join(parts) + "</p>"

def diff_badge(rel, repeat, diffs) -> str:
    out = []
    for base_label, lab, d in diffs or []:
        info = d.get(rel)
        if not info:
            continue
        ch = [k for k, _, _ in info["changed"] if k[2] == repeat]
        if ch:
            n = sum(1 for k in info["common"] if k[2] == repeat)
            out.append(f'<a class="chg-badge" href="{esc(rel)}.html#armset-diff">{esc(lab)} changed '
                       f'{len(ch)} of {n} cell{"s" if n != 1 else ""}</a>')
    return "".join(out)

def changes_section(rel, diffs, arm_set_label, short) -> list[str]:
    if not diffs:
        return []
    P = []
    for base_label, lab, d in diffs:
        info = d.get(rel)
        P.append(f'<h2 id="armset-diff">What {esc(lab)} changed</h2>')
        if not info or not info["common"]:
            P.append(f'<p class="note">This relation has no cell in both the {esc(base_label)} and the '
                     f'{esc(lab)} arm sets.</p>')
            continue
        rem = {v for v in info["removed"].values()}
        removed_all = sorted({w for v in rem for w in v[0]})
        rule = sorted({f"{v[1]} (k = {v[2]})" for v in rem if v[1]})
        rm = (f'{esc(", ".join(rule))} removed {len(removed_all)} label{"s" if len(removed_all) != 1 else ""} from '
              f'this relation&rsquo;s usable pool' + (" (the list varies by model; union shown)" if len(rem) > 1
                                                      else "")
              + ((f': {esc(", ".join(removed_all))}' if len(removed_all) <= 12 else
                  f' (<details class="exl"><summary>list them</summary>{esc(", ".join(removed_all))}</details>)')
                 if removed_all else "")) if rule else ""
        missing = ""
        if info["only_base"]:
            ms = sorted({short(k[0]) for k in info["only_base"]})
            missing = (f' Not yet in {esc(lab)}: {len(info["only_base"])} cell{"s" if len(info["only_base"]) != 1 else ""}'
                       f' ({esc(", ".join(ms))}).')
        if not info["changed"]:
            P.append(f'<p class="note nochg">{esc(lab)} changed <b>none</b> of this relation&rsquo;s '
                     f'{len(info["common"])} cells shared with {esc(base_label)}'
                     + (f'. {rm}, but none of them had been drawn here' if removed_all else "")
                     + "." + missing + "</p>")
            continue
        n = len(info["changed"])
        same = []
        if info["splits_same"]:
            same.append("subjects and splits are unchanged")
        if info["class_words_same"]:
            same.append("the PRE-REL classes are unchanged")
        P.append(f'<p class="note chg-note">{esc(lab)} changed <b>{n}</b> of the {len(info["common"])} cells this '
                 f'relation shares with {esc(base_label)}' + (f'. {rm}, and at least one of them had been drawn '
                                                               'here, so the labels were drawn again' if rm else "")
                 + (f'; {" and ".join(same)}: only the class &rarr; label map differs' if same else "")
                 + "." + missing + "</p>")

        groups = OrderedDict()
        for (m, r, rp), old, new in info["changed"]:
            oc = {c["pre_rel"]: c for c in old["classes"]}
            rows = []
            for c in new["classes"]:
                o = oc.get(c["pre_rel"])
                rows.append((c["pre_rel"], c["pre_rel_first_token"], c.get("pre_rel_skipped_prefix") or "", c["class"],
                             o and o["new_rel"], o and o["new_rel_first_token"], o and o["new_rel_token_id"],
                             c["new_rel"], c["new_rel_first_token"], c["new_rel_token_id"]))
            gone = tuple(sorted(set(oc) - {c["pre_rel"] for c in new["classes"]}))
            groups.setdefault((tuple(rows), gone), []).append(f"r{rp} {short(m)}")
        for (rows, gone), cells_ in groups.items():
            n_diff = sum(1 for r in rows if r[4] != r[7])
            T = [f'<div class="chg-group"><div class="fg-m">{esc(", ".join(cells_))}</div>',
                 f'<div class="scroll"><table class="chg"><thead><tr><th>PRE-REL</th><th>{esc(base_label)}</th>'
                 f'<th>{esc(lab)}</th></tr></thead><tbody>']
            for pre, pft, psk, pid, on, oft, oid, nn, nft, nid in rows:
                cls = ' class="d"' if on != nn else ""
                T.append(f'<tr{cls}><td><span class="tw pre">{tw(pre, pft, psk, pid)}</span></td>'
                         f'<td><span class="tw new">{tw(on, oft, None, oid) if on else "&mdash;"}</span></td>'
                         f'<td><span class="tw new">{tw(nn, nft, None, nid)}</span></td></tr>')
            T.append("</tbody></table></div>")
            T.append(f'<p class="muted fg-t">{n_diff} of {len(rows)} classes relabelled (highlighted rows)'
                     + (f'; classes only in {esc(base_label)}: {esc(", ".join(gone))}' if gone else "") + ".</p></div>")
            P.append("".join(T))
    return P

def write_html(outdir: Path, arm_set, prefix, cells, index_rows, summaries, mf, inv_root: Path,
               label=None, up="", nav=None, diffs=None):
    outdir.mkdir(parents=True, exist_ok=True)
    if not up:
        (outdir / "newrels.css").write_text(NEWRELS_CSS, "utf-8")
    rels = sorted({c["relation"] for c, *_ in cells})
    content_of = {(c["model"], c["relation"], c["repeat"]): (c, ct, p) for c, ct, p, _ in cells}
    summ = {(s["relation"], s["repeat"]): s for s in summaries}
    short = lambda m: m.split("/")[-1]
    head = lambda title: (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{esc(title)}</title>\n<link rel="stylesheet" href="{up}styles.css">\n'
        f'<link rel="stylesheet" href="{up}newrels.css">\n</head><body><main>\n')

    n_ident = mf["n_relation_repeats_bijection_identical_across_models"]
    H = [head("NEW-REL relations")]
    H.append("<h1>NEW-REL relations, materialized</h1>")
    H.append(nav_html(nav, arm_set, ""))
    for base_label, lab, d in diffs or []:
        ch = sorted(r for r, i in d.items() if i["changed"])
        if ch:
            H.append(f'<p class="note chg-note">{esc(lab)} changed cells in {len(ch)} relation'
                     f'{"s" if len(ch) != 1 else ""} (marked below; everything else is identical to '
                     f'{esc(base_label)}): ' + ", ".join(f'<a href="{esc(r)}.html#armset-diff">{esc(r)}</a>'
                                                          for r in ch) + ".</p>")
    H.append(f'<p class="note">Every fictional relation (NEW-REL) the fleet fine-tuned on, per model, relation '
             f'and repeat, read from the packed arms <strong>as run</strong> (arm set <code>{esc(arm_set)}</code>: '
             f'build commit <code>{esc(mf["build_commits"][0][:7])}</code>, label pool '
             f'<code>{esc(mf["label_pools"][0]["sha256"][:8])}</code>, {mf["label_pools"][0]["n"]} animals). '
             f'A later collision rule produces a new arm set and a new page. Each cell is also a '
             f'downloadable JSON5 file; <a href="{esc(prefix)}/{esc(arm_set)}/index.json5">index.json5</a> and '
             f'<a href="{esc(prefix)}/{esc(arm_set)}/manifest.json5">manifest.json5</a> describe the whole set.</p>')
    H.append('<div class="stat-grid">')
    H.append(f'<div class="stat"><strong>{mf["n_cells"]}</strong><span>cells (model &times; relation &times; repeat)</span></div>')
    H.append(f'<div class="stat"><strong>{mf["n_relations"]}</strong><span>relations, {mf["n_models"]} models</span></div>')
    H.append(f'<div class="stat"><strong>{n_ident} / {mf["n_relation_repeats_with_several_models"]}</strong><span>(relation, repeat) '
             f'pairs run on several models whose class &rarr; label map is identical across them</span></div>')
    H.append("</div>")
    H.append('<h2>Why the map differs by model</h2><p class="note">A PRE-REL class is a <em>token id</em>: the '
             'first non-degenerate token of the answer under that model\'s tokenizer. The bijection zips the classes '
             'in ascending token id with a keyed shuffle of the drawn animals, so the same animals land on different '
             'classes when two tokenizers order the classes differently, and a different number of retained classes '
             'draws a different animal subset. Models that share a tokenizer share the map.</p>')
    H.append('<p class="note">Rows: one per (relation, repeat). &ldquo;Map groups&rdquo; counts distinct class &rarr; '
             'label maps across models; open a row for the maps themselves. Each relation&rsquo;s page shows the '
             'held-out subjects of every class and where the source rows went on the way to the final subject '
             'count.</p>')
    H.append(TOKEN_LEGEND)
    H.append('<input id="q" type="search" placeholder="filter relations" class="filter">')
    H.append('<div class="scroll"><table class="rel-index"><thead><tr><th>relation</th><th>repeat</th><th>type</th><th>models</th>'
             '<th>classes</th><th>map groups</th><th>subject groups</th></tr></thead><tbody>')
    for s in summaries:
        rows = [r for r in index_rows if r["relation"] == s["relation"] and r["repeat"] == s["repeat"]]
        c0, ct0, _ = content_of[(rows[0]["model"], s["relation"], s["repeat"])]
        ncs = sorted({r["n_classes"] for r in rows})
        det = []
        for g in s["bijection_groups"]:
            sub = OrderedDict()
            for m in g:
                ct = content_of[(m, s["relation"], s["repeat"])][1]
                sig = tuple(sorted((c["pre_rel"], c["pre_rel_first_token"], c.get("pre_rel_skipped_prefix") or "",
                                    c["new_rel"], c["new_rel_first_token"]) for c in ct["classes"]))
                sub.setdefault(sig, []).append(m)
            subs = sorted(sub.values(), key=lambda ms: -len(ms))
            for vi, ms in enumerate(subs):
                ct = content_of[(ms[0], s["relation"], s["repeat"])][1]
                note = (f' <span class="muted">(same map, first-token variant {chr(97 + vi)} of {len(subs)})</span>'
                        if len(subs) > 1 else "")
                det.append(f'<div class="mapgroup"><div class="models">{esc(", ".join(short(m) for m in ms))}{note}</div>'
                           + '<div class="map">' + " ".join(
                               f'<span class="pair"><span class="pre">{tw(c["pre_rel"], c["pre_rel_first_token"], c.get("pre_rel_skipped_prefix"), c["class"])}</span>'
                               f'&rarr;<span class="new">{tw(c["new_rel"], c["new_rel_first_token"], None, c["new_rel_token_id"])}</span></span>'
                               for c in ct["classes"]) + "</div></div>")
        H.append(f'<tr class="rrow" data-rel="{esc(s["relation"])} {esc(shown_rel_name(ct0["relation"]["display_name"]))}">'
                 f'<td><a href="{esc(s["relation"])}.html">{esc(shown_rel_name(ct0["relation"]["display_name"]))}</a>'
                 f'<div class="slug">{esc(s["relation"])}</div>{diff_badge(s["relation"], s["repeat"], diffs)}</td><td>r{s["repeat"]}</td>'
                 f'<td>{esc(set_name(ct0["relation"]["relation_type"]))}</td><td>{s["n_models"]}</td>'
                 f'<td>{ncs[0]}{"" if len(ncs) == 1 else "&ndash;" + str(ncs[-1])}</td>'
                 f'<td>{len(s["bijection_groups"])}</td><td>{len(s["subject_content_groups"])}</td></tr>'
                 f'<tr class="drow" data-rel="{esc(s["relation"])} {esc(shown_rel_name(ct0["relation"]["display_name"]))}">'
                 f'<td colspan="7"><details><summary>class &rarr; label maps</summary>{"".join(det)}</details></td></tr>')
    H.append("</tbody></table></div>")
    H.append("<script>const q=document.getElementById('q');q.addEventListener('input',()=>{const v=q.value.toLowerCase();"
             "document.querySelectorAll('tr[data-rel]').forEach(t=>{t.style.display=t.dataset.rel.toLowerCase().includes(v)?'':'none'})});</script>")
    H.append("</main></body></html>\n")
    (outdir / "index.html").write_text("\n".join(H), "utf-8")

    for rel in rels:
        rows = [r for r in index_rows if r["relation"] == rel]
        c0, ct0, _ = content_of[(rows[0]["model"], rel, rows[0]["repeat"])]
        info = ct0["relation"]

        models = sorted({r["model"] for r in rows})
        cov = defaultdict(dict)
        lowest = {}
        for m in models:
            rp = min(r["repeat"] for r in rows if r["model"] == m)
            _, ct, pv = content_of[(m, rel, rp)]
            lowest[m] = (ct, pv)
            for s in ct["subjects"]:
                cov[s["subject"]][m] = s["split"][0].upper()
        cells_js = [OrderedDict([("model", r["model"]), ("repeat", r["repeat"]),
                                 ("path", f'{prefix}/{arm_set}/{r["path"]}'), ("n_classes", r["n_classes"]),
                                 ("n_train", r["n_train_subjects"]), ("n_heldout", r["n_heldout_subjects"]),
                                 ("flags", r["n_flags_pre_rel"] + r["n_flags_new_rel"])])
                    for r in sorted(rows, key=lambda r: (r["repeat"], r["model"]))]
        sums = [summ[k] for k in sorted(summ) if k[0] == rel]
        P = [head(shown_rel_name(info["display_name"]))]
        P.append(f'<p class="crumb"><a href="index.html">&larr; all NEW-REL relations</a></p>')
        P.append(nav_html(nav, arm_set, f"{rel}.html"))
        P.append(f'<h1>{esc(shown_rel_name(info["display_name"]))}</h1>')
        P.append(f'<p class="note"><code>{esc(rel)}</code> &middot; {esc(set_name(info["relation_type"]))} &middot; subject noun '
                 f'&ldquo;{esc(info["subject_noun"])}&rdquo;'
                 + (f' &middot; PRE-REL source: {esc(shown_pre_rel_source(info["pre_rel_source"]))}' if info["pre_rel_source"] else "")
                 + "</p>")
        if info["pre_rel_prompt_templates"]:
            P.append('<p class="note">PRE-REL prompt: ' + " / ".join(f"<code>{esc(t)}</code>" for t in info["pre_rel_prompt_templates"])
                     + (f' ({esc(info["pre_rel_domain"])} &rarr; {esc(info["pre_rel_range"])})' if info["pre_rel_domain"] else "")
                     + "</p>")
        if info["newrel_prompt_notes"]:
            P.append(f'<details class="note"><summary>Template author&rsquo;s notes</summary><p>{esc(info["newrel_prompt_notes"])}</p></details>')
        P.append('<p class="note toc">On this page: <a href="#cells">cells</a> &middot; '
                 + ('<a href="#armset-diff">arm-set changes</a> &middot; ' if diffs else '')
                 + '<a href="#funnel">where the '
                 'subjects went</a> &middot; <a href="#maps">class maps and held-out subjects</a> &middot; '
                 '<a href="#one">one cell</a> &middot; <a href="#coverage">subject coverage</a></p>')
        P.append('<h2 id="cells">Cells</h2>')
        P.append('<div class="scroll"><table class="cells"><thead><tr><th>repeat</th><th>model</th><th>classes</th><th>train</th>'
                 '<th>heldout</th><th>map group</th><th>subject group</th><th>token flags</th><th>JSON5</th></tr></thead><tbody>')
        for s in sums:
            bg = {m: i + 1 for i, g in enumerate(s["bijection_groups"]) for m in g}
            sg = {m: i + 1 for i, g in enumerate(s["subject_content_groups"]) for m in g}
            for r in [x for x in rows if x["repeat"] == s["repeat"]]:
                P.append(f'<tr><td>r{r["repeat"]}</td><td>{esc(short(r["model"]))}</td><td>{r["n_classes"]}</td>'
                         f'<td>{r["n_train_subjects"]}</td><td>{r["n_heldout_subjects"]}</td>'
                         f'<td>{bg[r["model"]]}</td><td>{sg[r["model"]]}</td>'
                         f'<td>{r["n_flags_pre_rel"] + r["n_flags_new_rel"]}</td>'
                         f'<td><a href="{esc(prefix)}/{esc(arm_set)}/{esc(r["path"])}" download>r{r["repeat"]}.json5</a></td></tr>')
        P.append("</tbody></table></div>")
        P.append('<p class="note">Same map group = identical PRE-REL word &rarr; NEW-REL label map. Same subject group = '
                 'byte-identical subject lines (subject, class, labels, split).</p>')
        P.extend(changes_section(rel, diffs, label, short))
        P.extend(funnel_section(rel, models, lowest, inventory_for(inv_root, rel), inv_root, short))
        P.extend(maps_section(rel, sums, content_of, short))
        P.append('<h2 id="one">One cell</h2><label class="sel">Cell: <select id="cell">' + "".join(
            f'<option value="{i}">r{c["repeat"]} &middot; {esc(short(c["model"]))}</option>' for i, c in enumerate(cells_js))
            + '</select></label><div id="view"><p class="note">Loading&hellip;</p></div>')
        P.append('<h2 id="coverage">Subject coverage across models</h2>')
        P.append('<p class="note">Split of every subject in each model&rsquo;s arm (lowest repeat; a repeat changes only '
                 'the labels, not the splits): <b>T</b> train, <b>H</b> heldout, <b>S</b> support, blank = not in that '
                 'model&rsquo;s arm (its tokenizer merged or excluded the class, or admission dropped the subject).</p>')
        P.append('<input id="cq" type="search" placeholder="filter subjects" class="filter">')
        P.append('<div class="scroll"><table class="cov"><thead><tr><th>subject</th>' + "".join(
            f'<th class="m">{esc(short(m))}</th>' for m in models) + "</tr></thead><tbody>")
        for subj in sorted(cov):
            P.append(f'<tr><td>{esc(subj)}</td>' + "".join(
                f'<td class="c {cov[subj].get(m, "")}">{cov[subj].get(m, "")}</td>' for m in models) + "</tr>")
        P.append("</tbody></table></div>")
        P.append("<script>const CELLS=" + json.dumps(cells_js, ensure_ascii=False) + ";\n" + VIEW_JS + "</script>")
        P.append("</main></body></html>\n")
        (outdir / f"{rel}.html").write_text("\n".join(P), "utf-8")
    print(f"HTML wrote index.html{'' if up else ', newrels.css'} and {len(rels)} relation pages to {outdir}")

NEWRELS_CSS = """/* NEW-REL materialized pages: additions on top of styles.css (copied from
   ../fleet-smoke), using its colour tokens only, so light and dark mode follow
   styles.css.  Written by linroute scripts/materialize_newrels.py; do not edit
   here. */
main { max-width: 100%; }
.crumb { font-size: .84rem; margin: 0 0 .4rem; }
.slug { color: var(--text-muted); font-size: .74rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.muted { color: var(--text-muted); }
input.filter, .sel select, #view select {
  font: inherit; font-size: .84rem; padding: .25rem .45rem; border-radius: 6px;
  border: 1px solid var(--border-strong); background: var(--surface-2); color: var(--text);
  max-width: 100%;
}
input.filter { width: 16rem; margin: .3rem 0 .5rem; }
.sel { font-size: .88rem; display: inline-block; margin: .2rem 0 .6rem; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: .3rem 0 .6rem; }
.scroll table { min-width: 0; }
table.rel-index td, #view td { white-space: normal; text-align: left; vertical-align: top; }
table.rel-index th, #view th { text-align: left; }
table.rel-index { min-width: 0; }
table.rel-index tr.drow td { border-bottom: 1px solid var(--border-strong); padding-top: 0; }
table.rel-index tr.rrow td { border-bottom: 0; }
details summary { cursor: pointer; color: var(--text-2); font-size: .82rem; }
.mapgroup { margin: .35rem 0 .6rem; }
.mapgroup .models { font-size: .76rem; color: var(--text-muted); margin-bottom: .15rem; }
.map { display: flex; flex-wrap: wrap; gap: .15rem .7rem; font-size: .8rem; }
.pair { white-space: nowrap; }
.pair .new { font-weight: 600; }
table.cells { min-width: 0; }
table.cov { min-width: 0; font-size: .74rem; }
table.cov th.m { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; font-weight: 500; }
table.cov td.c { text-align: center; padding: .1rem .3rem; font-variant-numeric: tabular-nums; }
table.cov td.T { color: var(--series-1); }
table.cov td.H { color: var(--series-2); font-weight: 650; }
table.cov td.S { color: var(--text-muted); }
.sp.heldout { color: var(--series-2); font-weight: 600; }
.sp.support { color: var(--text-muted); }
@media (max-width: 480px) { input.filter { width: 100%; } .pair { white-space: normal; } }
.note code { white-space: normal; overflow-wrap: anywhere; }
table.cells th, table.cells td { text-align: left; }

/* first token of a word: the marked part is the token the loss reads, the rest is muted */
.tw, .pair .pre, .pair .new { color: var(--text-muted); }
.tw u, .pair u, u.lg {
  text-decoration: none; color: var(--text);
  background: color-mix(in srgb, var(--series-1) 16%, transparent);
  border-bottom: 2px solid var(--series-1); border-radius: 2px 2px 0 0; padding: 0 1px;
}
.ftb {
  display: inline-block; margin-left: .25em; padding: 0 .3em; border: 1px solid var(--border-strong);
  border-radius: 4px; background: var(--surface-2); color: var(--text-2); font-weight: 500;
  font: .72em/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; vertical-align: 1px;
}
.tw u.pb, .pair u.pb, u.pb { border-bottom: 2px dotted var(--series-1); background: color-mix(in srgb, var(--series-1) 9%, transparent); }
.pbn { font-size: .62em; color: var(--text-muted); margin-left: .15em; font-weight: 500; white-space: nowrap; }
.tok-legend { font-size: .8rem; color: var(--text-2); margin: .3rem 0 .6rem; }
.toc { font-size: .82rem; }

/* class cards: one per class, held-out boxed, train folded, support apart */
details.mg { margin: .4rem 0 .7rem; border-top: 1px solid var(--border); padding-top: .35rem; }
details.mg > summary { font-size: .86rem; color: var(--text); overflow-wrap: anywhere; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 15rem), 1fr)); gap: .5rem; margin: .45rem 0 .5rem; }
.ccard { border: 1px solid var(--border); border-radius: 8px; background: var(--surface); padding: .45rem .6rem; min-width: 0; }
.cmap { font-size: .95rem; margin-bottom: .3rem; overflow-wrap: anywhere; }
.cmap .new { font-weight: 650; }
.cmap .arr { color: var(--text-muted); }
.lab { display: block; font-size: .66rem; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); margin-bottom: .1rem; }
.hbox {
  border: 1.5px solid var(--series-2); border-radius: 6px; padding: .2rem .4rem; margin: .25rem 0;
  background: color-mix(in srgb, var(--series-2) 9%, transparent); overflow-wrap: anywhere;
}
.hbox .lab { color: var(--text); font-weight: 650; }
.hbox.none { border-style: dashed; font-size: .78rem; color: var(--text-muted); }
.chip { display: inline-block; margin: .08rem .3rem .08rem 0; font-size: .86rem; font-weight: 600; color: var(--text); }
.tbox { font-size: .78rem; color: var(--text-2); overflow-wrap: anywhere; margin: .2rem 0; }
.tbox summary { font-size: .74rem; }
.sbox { border: 1px dashed var(--border-strong); border-radius: 6px; padding: .15rem .4rem; margin: .25rem 0; font-size: .76rem; color: var(--text-muted); overflow-wrap: anywhere; }
.vm { font-size: .7rem; color: var(--text-muted); margin-top: .35rem; border-top: 1px solid var(--grid); padding-top: .25rem; overflow-wrap: anywhere; }

/* subject funnel */
.fg { border-left: 3px solid var(--series-3); background: var(--surface); padding: .3rem .7rem; margin: .5rem 0; }
.fg-m { font-size: .8rem; color: var(--text-2); font-weight: 600; overflow-wrap: anywhere; }
.fg ol { margin: .25rem 0 .2rem 1.3rem; padding: 0; font-size: .86rem; }
.fg li { margin: .1rem 0; overflow-wrap: anywhere; }
.fg details.exl, .chg-note details.exl { display: inline; }
.fg-t { font-size: .78rem; margin: .1rem 0 .2rem; }
.warn-t { color: var(--warn-text); }

/* arm-set switch and what a later arm set changed */
.armset { font-size: .84rem; margin: .2rem 0 .6rem; overflow-wrap: anywhere; }
.armset code { font-size: .74rem; }
.chg-badge {
  display: inline-block; margin-top: .2rem; font-size: .7rem; font-weight: 600; padding: .05rem .4rem;
  border-radius: 999px; border: 1px solid var(--warn-border); background: var(--warn-bg); color: var(--warn-text);
}
.chg-note { border-left: 3px solid var(--warn-border); background: var(--warn-bg); color: var(--warn-text); padding: .3rem .6rem; }
.chg-note a { color: inherit; }
.nochg { border-left: 3px solid var(--series-3); padding: .2rem .6rem; }
.chg-group { margin: .5rem 0 .8rem; }
table.chg { min-width: 0; font-size: .86rem; }
table.chg th, table.chg td { text-align: left; }
table.chg tr.d td { background: color-mix(in srgb, var(--warn-border) 14%, transparent); }
"""

VIEW_JS = r"""
const esc=s=>s==null?'':String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const strip=t=>t.split('\n').filter(l=>!l.trim().startsWith('//')).join('\n');
const SP='␣';
// twin of tw() in materialize_newrels.py: whole word, first token marked, else a badge
function tw(w,ft,sk,tid){w=w==null?'':String(w); if(!ft) return esc(w);
  const core=ft.split(SP).join(' ').replace(/^ +/,''), s=(sk||'').split(SP).join(' ').replace(/^ +/,''), p=s+core;
  if(core&&w.slice(0,p.length).toLowerCase()===p.toLowerCase()) return esc(w.slice(0,s.length))+'<u>'+esc(w.slice(s.length,p.length))+'</u>'+esc(w.slice(p.length));
  if(core&&/^\uFFFD+$/.test(core)&&w.length>s.length){const i=s.length, n='first token is a partial UTF-8 byte of this letter'+(tid!=null?' (token id '+tid+')':'');
    return esc(w.slice(0,i))+'<u class="pb" title="'+esc(n)+'">'+esc(w[i])+'</u>'+esc(w.slice(i+1))+'<sup class="pbn" title="'+esc(n)+'">partial byte'+(tid!=null?' '+tid:'')+'</sup>';}
  return esc(w)+'<span class="ftb">'+esc(ft)+'</span>';}
const twp=(w,ft,sk,cls,tid)=>'<span class="tw'+(cls?' '+cls:'')+'">'+tw(w,ft,sk,tid)+'</span>';
const sel=document.getElementById('cell'), view=document.getElementById('view');
const cache={};
function table(head,rows){return '<div class="scroll"><table><thead><tr>'+head.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function cardBody(h,t,s){let o=h.length?'<div class="hbox"><span class="lab">held-out ('+h.length+')</span>'+h.map(x=>'<span class="chip h">'+esc(x)+'</span>').join('')+'</div>':'<div class="hbox none"><span class="lab">held-out</span>none</div>';
  if(t.length) o+='<details class="tbox"><summary>train ('+t.length+')</summary>'+esc(t.join(', '))+'</details>';
  if(s.length) o+='<div class="sbox"><span class="lab">support: few-shot only, in no NEW-REL row</span>'+esc(s.join(', '))+'</div>';
  return o;}
function show(d,c){
  const cls={}, exc={}; d.classes.forEach(x=>cls[x['class']]=x); d.excluded_classes.forEach(x=>exc[x['class']]=x);
  const by={}; d.subjects.forEach(s=>{const b=by[s['class']]=by[s['class']]||{train:[],heldout:[],support:[]}; b[s.split].push(s.subject);});
  let h='<p class="note"><a href="'+esc(c.path)+'" download>Download this cell as JSON5</a> &middot; '+esc(d.cell.model)+' &middot; r'+d.cell.repeat+' &middot; '+d.classes.length+' classes &middot; '+d.provenance.n_train_subjects+' train / '+d.provenance.n_heldout_subjects+' heldout / '+d.provenance.n_support_subjects+' support subjects</p>';
  h+='<h3>Classes and their held-out subjects</h3><div class="cards">'+d.classes.map(x=>{const b=by[x['class']]||{train:[],heldout:[],support:[]};
    return '<div class="ccard"><div class="cmap">'+twp(x.pre_rel,x.pre_rel_first_token,x.pre_rel_skipped_prefix,'pre',x['class'])+' <span class="arr">&rarr;</span> '+twp(x.new_rel,x.new_rel_first_token,null,'new',x.new_rel_token_id)+'</div>'+cardBody(b.heldout,b.train,b.support)+'</div>';}).join('')+'</div>';
  const orph=d.subjects.filter(s=>s.split==='support'&&!cls[s['class']]);
  if(orph.length) h+='<p class="note sbox">Support subjects of excluded classes (no NEW-REL label): '+esc(orph.map(s=>s.subject+' ('+s.pre_rel+')').join(', '))+'</p>';
  h+='<h3>Templates</h3>'+table(['name','role','template','example (first training subject)'],d.templates.map(t=>[esc(t.name),esc(t.role),'<code>'+esc(t.template)+'</code>','<code>'+esc(t.example_prompt)+'</code><b>'+esc(t.example_target)+'</b>']));
  h+='<h3>Class &rarr; label map</h3><p class="note">class = PRE-REL token id; first tokens are decoded with this model&rsquo;s tokenizer, &#9251; = space.</p>';
  h+=table(['class','PRE-REL','first token','NEW-REL','token id','first token','tokens','train','heldout','notes'],d.classes.map(x=>[x['class'],twp(x.pre_rel,x.pre_rel_first_token,x.pre_rel_skipped_prefix,null,x['class']),'<code>'+esc(x.pre_rel_first_token)+'</code>','<b>'+twp(x.new_rel,x.new_rel_first_token,null,null,x.new_rel_token_id)+'</b>',x.new_rel_token_id,'<code>'+esc(x.new_rel_first_token)+'</code>',x.new_rel_n_tokens,x.n_train,x.n_heldout,esc((x.pre_rel_members?'members: '+x.pre_rel_members.join(', '):'')+(x.pre_rel_skipped_prefix?' skipped prefix '+x.pre_rel_skipped_prefix:''))]));
  const tf=d.token_flags;
  if(tf.pre_rel.length||tf.new_rel.length){h+='<h3>Token flags</h3>'+table(['side','class','word','first token','flag'],tf.pre_rel.map(f=>['PRE-REL',f['class'],twp(f.pre_rel,f.first_token,null,null,f['class']),'<code>'+esc(f.first_token)+'</code>',esc((f.prefix_of_other_classes_words.length?'prefix of: '+f.prefix_of_other_classes_words.join(', '):'')+(f.n_merged_members>1?' merges '+f.n_merged_members+' answers':''))]).concat(tf.new_rel.map(f=>['NEW-REL',f['class'],twp(f.new_rel,f.first_token,null,null,cls[f['class']]?cls[f['class']].new_rel_token_id:null),'<code>'+esc(f.first_token)+'</code>','prefix of: '+esc(f.prefix_of_other_labels.join(', '))])));}
  if(d.excluded_classes.length){h+='<h3>Excluded PRE-REL classes</h3>'+table(['class','PRE-REL','first token','reason','support'],d.excluded_classes.map(x=>[x['class'],twp(x.pre_rel,x.pre_rel_first_token,null,null,x['class']),'<code>'+esc(x.pre_rel_first_token)+'</code>',esc(x.reason),x.support]));}
  h+='<h3>Subjects</h3><p class="note"><select id="sp"><option value="">all splits</option><option>train</option><option>heldout</option><option>support</option></select> <input id="sq" type="search" placeholder="filter" class="filter"></p><div id="subs"></div>';
  view.innerHTML=h;
  const preW=s=>{const k=cls[s['class']]||exc[s['class']]; const w=s.pre_rel_object||s.pre_rel; return k&&!s.pre_rel_object?twp(w,k.pre_rel_first_token,k.pre_rel_skipped_prefix,null,s['class']):esc(s.pre_rel)+(s.pre_rel_object?' <span class="muted">('+esc(s.pre_rel_object)+')</span>':'');};
  const newW=s=>{const k=cls[s['class']]; return s.new_rel==null?'<span class="muted">none</span>':'<b>'+(k?twp(s.new_rel,k.new_rel_first_token,null,null,k.new_rel_token_id):esc(s.new_rel))+'</b>';};
  const draw=()=>{const sp=document.getElementById('sp').value, q=document.getElementById('sq').value.toLowerCase();
    const rs=d.subjects.filter(s=>(!sp||s.split===sp)&&(!q||(s.subject+' '+s.pre_rel+' '+(s.new_rel||'')).toLowerCase().includes(q)));
    document.getElementById('subs').innerHTML=table(['subject','class','PRE-REL','NEW-REL','split'],rs.map(s=>[esc(s.subject),s['class'],preW(s),newW(s),'<span class="sp '+s.split+'">'+s.split+'</span>']));};
  document.getElementById('sp').onchange=draw; document.getElementById('sq').oninput=draw; draw();
}
function load(){const c=CELLS[sel.value]; if(cache[c.path]) return show(cache[c.path],c);
  view.innerHTML='<p class="note">Loading&hellip;</p>';
  fetch(c.path).then(r=>{if(!r.ok) throw new Error(r.status); return r.text();}).then(t=>{cache[c.path]=JSON.parse(strip(t)); show(cache[c.path],c);}).catch(e=>{view.innerHTML='<p class="note">Could not load '+esc(c.path)+': '+esc(e)+'</p>';});}
sel.onchange=load; load();
const cq=document.getElementById('cq'); cq.addEventListener('input',()=>{const v=cq.value.toLowerCase(); document.querySelectorAll('table.cov tbody tr').forEach(t=>{t.style.display=t.firstChild.textContent.toLowerCase().includes(v)?'':'none'})});
"""

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print(f"MATERIALIZE_FAIL {e}", file=sys.stderr)
        sys.exit(2)
