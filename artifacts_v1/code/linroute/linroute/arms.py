from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from linroute import admission as admission_rules
from linroute import spans
from linroute.prompts import check_template

__all__ = [
    "SCHEMA",
    "MANIFEST_SCHEMA",
    "GENERATOR",
    "GENERATOR_VERSION",
    "ROSTER_TOKENIZERS",
    "REPEATS",
    "TARGET_FORMAT",
    "ENTITY_GROUPING",
    "N_TRAIN_TEMPLATES",
    "N_GATE_TEMPLATES",
    "MAX_CLASSES",
    "LABEL_POOL_SCHEMA",
    "LABEL_RULE",
    "LABEL_EXCLUSION_SCHEMA",
    "Paraphrase",
    "NewRelPrompts",
    "AnimalAssertionError",
    "ArmBuildError",
    "load_newrel",
    "pool_category_word",
    "require_category_slot",
    "load_animals",
    "load_label_exclusion",
    "load_manifest",
    "retained_classes",
    "manifest_subjects",
    "class_representatives",
    "keyed_shuffle",
    "choose_animals",
    "first_tokens",
    "label_pool",
    "class_bijection",
    "make_records",
    "mapping_rows",
    "assert_animals",
    "arm_clauses",
    "validate_arm",
    "CHECKS_SCHEMA",
    "CHECKS_VERSION",
    "arm_name",
    "build_arm",
    "write_arm",
    "sha256_file",
]

SCHEMA = "linroute/prompts/newrel/1"

MANIFEST_SCHEMA = admission_rules.SCHEMA
MANIFEST_SCHEMA_V1 = admission_rules.SCHEMA_V1

GENERATOR = "scripts/make_relation_arm.py"

GENERATOR_VERSION = "2.3.0"

LABEL_EXCLUSION_SCHEMA = "linroute/label-exclusion/1"

ROSTER_TOKENIZERS = ("allenai/Olmo-3-1025-7B", "Qwen/Qwen3.5-9B")

REPEATS = (0, 1, 2)

TARGET_FORMAT = " {animal}"

CATEGORY_SLOT = "{category}"

CURATED_CATEGORY_WORD = "animal"

ENTITY_GROUPING = "subject_string"

N_TRAIN_TEMPLATES = 4
N_GATE_TEMPLATES = 1

MAX_CLASSES = 78

LABEL_POOL_SCHEMA = "linroute/label-pool/1"

LABEL_RULE = "first_token"

CHECKS_SCHEMA = "linroute/arm-checks/1"

CHECKS_VERSION = "1.0.0"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

class ArmBuildError(ValueError):
    pass

class AnimalAssertionError(ArmBuildError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report

@dataclass(frozen=True)
class Paraphrase:
    name: str
    template: str

@dataclass(frozen=True)
class NewRelPrompts:
    relation: str
    relation_slug: str
    relation_type: str
    subject_noun: str
    train: tuple[Paraphrase, ...]
    gate: tuple[Paraphrase, ...]
    target: str
    rubric_version: str
    author: str
    notes: str
    path: str = ""
    sha256: str = ""

    category_word: str = ""

    @property
    def has_category_slot(self) -> bool:
        return any(CATEGORY_SLOT in p.template for p in self.train + self.gate)

    def with_category(self, word: str) -> "NewRelPrompts":
        if not isinstance(word, str) or not word.strip():
            raise ArmBuildError(
                f"{self.path or self.relation_slug}: the label pool declared no "
                f"category word ({word!r}), but "
                f"{sum(CATEGORY_SLOT in p.template for p in self.train + self.gate)}"
                f" of this file's {len(self.train) + len(self.gate)} paraphrases "
                f"carry a {CATEGORY_SLOT} slot.  Filling it with nothing would "
                f"render 'showing one .' and leaving it would render the literal "
                f"'{CATEGORY_SLOT}'; both train.  Declare `category_word` in the "
                f"pool's provenance.json."
            )
        if "{" in word or "}" in word:
            raise ArmBuildError(
                f"category word {word!r} contains a brace, which would survive "
                f"substitution and break the subject slot"
            )

        def fill(p: Paraphrase) -> Paraphrase:
            where = f"{Path(self.path).name or self.relation_slug}:{p.name}"

            return Paraphrase(
                p.name,
                check_template(p.template.replace(CATEGORY_SLOT, word), where),
            )

        return replace(
            self,
            train=tuple(fill(p) for p in self.train),
            gate=tuple(fill(p) for p in self.gate),
            category_word=word,
        )

    def as_template_map(self) -> dict[str, dict[str, str]]:
        return {
            "train": {p.name: p.template for p in self.train},
            "gate": {p.name: p.template for p in self.gate},
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.train + self.gate)

    def render_target(self, animal: str) -> str:
        return self.target.format(animal=animal)

def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_newrel(path: Path | str) -> NewRelPrompts:
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ArmBuildError(f"{path}: not a YAML mapping")
    where = path.name
    got = doc.get("schema")
    if got != SCHEMA:
        raise ArmBuildError(f"{where}: schema is {got!r}, expected {SCHEMA!r}")

    required = ("relation", "relation_slug", "subject_noun", "train", "gate",
                "target", "rubric_version", "author")
    missing = [k for k in required if k not in doc]
    if missing:
        raise ArmBuildError(f"{where}: missing key(s) {missing}")

    slug = str(doc["relation_slug"])
    if not _SLUG_RE.match(slug):
        raise ArmBuildError(
            f"{where}: relation_slug {slug!r} is not [a-z0-9][a-z0-9_]* -- it "
            f"names a directory and seeds two draws, so it must be stable and "
            f"path-safe"
        )

    def paraphrases(key: str, expected: int) -> tuple[Paraphrase, ...]:
        items = doc[key]
        if not isinstance(items, list) or len(items) != expected:
            n = len(items) if isinstance(items, list) else "not a list"
            raise ArmBuildError(
                f"{where}: {key} has {n} entries, expected exactly {expected}"
            )
        out = []
        for i, p in enumerate(items):
            if not isinstance(p, dict) or "name" not in p or "template" not in p:
                raise ArmBuildError(f"{where}: {key}[{i}] needs name and template")
            name, tpl = str(p["name"]), p["template"]
            if not name.strip():
                raise ArmBuildError(f"{where}: {key}[{i}] has an empty name")
            if "{animal}" in str(tpl):
                raise ArmBuildError(
                    f"{where}: {key}.{name} contains an '{{animal}}' slot.  A "
                    f"paraphrase is the QUESTION; only `target` carries the "
                    f"answer."
                )
            out.append(Paraphrase(
                name,

                check_template(tpl, f"{where}:{key}.{name}",
                               allow=(CATEGORY_SLOT,)),
            ))
        return tuple(out)

    train = paraphrases("train", N_TRAIN_TEMPLATES)
    gate = paraphrases("gate", N_GATE_TEMPLATES)
    names = [p.name for p in train + gate]
    if len(set(names)) != len(names):
        raise ArmBuildError(f"{where}: duplicate paraphrase names in {names}")

    target = doc["target"]
    if target != TARGET_FORMAT:
        raise ArmBuildError(
            f"{where}: target is {target!r}, expected {TARGET_FORMAT!r}.  The "
            f"leading space is what makes the answer one token, and the arm's "
            f"assertion is written against exactly this wording."
        )

    return NewRelPrompts(
        relation=str(doc["relation"]),
        relation_slug=slug,
        relation_type=str(doc.get("relation_type", "newrel")),
        subject_noun=str(doc["subject_noun"]),
        train=train,
        gate=gate,
        target=target,
        rubric_version=str(doc["rubric_version"]),
        author=str(doc["author"]),
        notes=str(doc.get("notes", "") or ""),
        path=str(path),
        sha256=sha256_file(path),
    )

def pool_category_word(path: Path | str) -> str:
    path = Path(path)
    if path.is_dir():
        path = path / "provenance.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArmBuildError(f"cannot read the pool declaration {path}: {exc}") from exc
    except ValueError as exc:
        raise ArmBuildError(f"{path} is not valid JSON: {exc}") from exc
    word = doc.get("category_word")
    if not isinstance(word, str) or not word.strip():
        raise ArmBuildError(
            f"{path} declares category_word {word!r}.  A pool must name the "
            f"noun its members are called, because every NEW-REL paraphrase "
            f"asks for 'one {{category}}' and there is no sensible default: "
            f"the curated animal pool's word is "
            f"{CURATED_CATEGORY_WORD!r} and every other pool's is its own."
        )
    return word

def require_category_slot(prompts: NewRelPrompts, word: str) -> None:
    if word == CURATED_CATEGORY_WORD:
        return
    missing = [p.name for p in prompts.train + prompts.gate if CATEGORY_SLOT not in p.template]
    if missing:
        raise ArmBuildError(
            f"{prompts.path or prompts.relation_slug}: paraphrase(s) {missing} "
            f"carry no {CATEGORY_SLOT} slot, but the label pool's category word "
            f"is {word!r}, not the curated {CURATED_CATEGORY_WORD!r}.  A "
            f"slotless paraphrase names its category in literal text, so its "
            f"prompt would ask for the wrong kind of thing while the arm trains "
            f"on {word!r} labels.  Put {CATEGORY_SLOT} where the file names the "
            f"category (docs/arms.md, \"The {{category}} slot\")."
        )

def load_animals(path: Path | str) -> list[str]:
    animals = [ln.strip() for ln in
               Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(set(animals)) != len(animals):
        raise ArmBuildError(f"{path}: contains duplicate animals")
    if not animals:
        raise ArmBuildError(f"{path}: empty animal pool")
    return animals

def load_label_exclusion(path: Path | str, pool_path: Path | str | None = None) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("schema") != LABEL_EXCLUSION_SCHEMA:
        raise ArmBuildError(f"{path}: schema {doc.get('schema')!r}, expected "
                            f"{LABEL_EXCLUSION_SCHEMA!r}")
    if pool_path is not None:
        got = sha256_file(pool_path)
        want = doc.get("pool", {}).get("sha256")
        if got != want:
            raise ArmBuildError(
                f"{path} was written for the pool with sha256 {want}, but the "
                f"pool given is {pool_path} with sha256 {got}"
            )
    doc["_path"] = str(path)
    doc["_sha256"] = hashlib.sha256(raw).hexdigest()
    return doc

def load_manifest(path: Path | str) -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    got = manifest.get("schema")
    if got != MANIFEST_SCHEMA:
        extra = ""
        if got == MANIFEST_SCHEMA_V1:
            extra = (f".  That is the {admission_rules.V1_HOLDOUT_RULE} protocol: it "
                     f"holds out one subject per class, so an arm built from it "
                     f"trains on a different T and is a different experiment.  "
                     f"Rebuild the manifest under {admission_rules.HOLDOUT_RULE}.")
        raise ArmBuildError(
            f"{path}: manifest schema is {got!r}, expected "
            f"{MANIFEST_SCHEMA!r}{extra}"
        )
    return manifest

def retained_classes(manifest: Mapping) -> list[int]:
    classes = [int(c["effective_class"]) for c in manifest["classes"]
               if c.get("retained")]
    if len(set(classes)) != len(classes):
        raise ArmBuildError("manifest lists a retained class twice")
    return sorted(classes)

def manifest_subjects(manifest: Mapping, role: str) -> list[dict]:
    if role not in ("train", "holdout", "support"):
        raise ArmBuildError(f"unknown subject role {role!r}")
    return [dict(s) for s in manifest[role]["subjects"]]

def class_representatives(
    records: Iterable[Mapping], classes: Sequence[int]
) -> dict[int, str]:
    wanted = set(int(c) for c in classes)
    seen: dict[int, str] = {}
    for row in records:
        if row.get("status") != "included":
            continue
        cls = row.get("effective_class")
        if cls is None:
            continue
        cls = int(cls)
        if cls in wanted and cls not in seen:
            seen[cls] = str(row["object"])
    missing = sorted(wanted - set(seen))
    if missing:
        raise ArmBuildError(
            f"the records file has no included record for retained class(es) "
            f"{missing}; the manifest and the records are not the same pool"
        )
    labels = {c: seen[c] for c in sorted(wanted)}
    collisions = {}
    for c, label in labels.items():
        collisions.setdefault(label, []).append(c)
    clashing = {k: v for k, v in collisions.items() if len(v) > 1}
    if clashing:
        raise ArmBuildError(
            f"two retained classes share a representative object: {clashing}.  "
            f"The `region` column would no longer identify the class."
        )
    return labels

def load_records(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def entity_grouping_note(records: Iterable[Mapping]) -> dict:
    statuses: dict[str, int] = {}
    resolved = 0
    for row in records:
        status = str(row.get("entity_status", "absent"))
        statuses[status] = statuses.get(status, 0) + 1
        if row.get("entity_id") is not None:
            resolved += 1
    return {
        "grouping": ENTITY_GROUPING,
        "entity_status_counts": dict(sorted(statuses.items())),
        "n_records_with_entity_id": resolved,
        "note": (
            "Subjects are grouped by their surface string only.  This arm makes "
            "no alias-aware claim even if the records carry entity ids."
        ),
    }

def keyed_shuffle(items: Sequence[str], key: str) -> list[str]:
    def sort_key(item: str) -> tuple[str, str]:
        blob = (key + "\x00" + item).encode("utf-8")
        return hashlib.sha256(blob).hexdigest(), item

    return sorted(items, key=sort_key)

def choose_animals(
    animals: Sequence[str], n: int, *, relation_slug: str, role_seed: int
) -> list[str]:
    if n < 0:
        raise ArmBuildError(f"cannot draw {n} animals")
    if n > len(animals):
        raise ArmBuildError(
            f"the relation has {n} retained classes but the curated pool holds "
            f"only {len(animals)} animals.  Cap the number of classes upstream "
            f"(the admission step decides which classes survive); this builder "
            f"will not silently drop one."
        )
    key = f"animals|{relation_slug}|{role_seed}"
    return keyed_shuffle(list(animals), key)[:n]

def class_bijection(
    classes: Sequence[int], subset: Sequence[str], *,
    relation_slug: str, repeat: int,
) -> dict[int, str]:
    if repeat not in REPEATS:
        raise ArmBuildError(f"repeat must be one of {list(REPEATS)}, got {repeat}")
    if len(classes) != len(subset):
        raise ArmBuildError(
            f"{len(classes)} classes vs {len(subset)} animals: not a bijection"
        )
    permuted = keyed_shuffle(list(subset), f"bijection|{relation_slug}|{repeat}")
    return dict(zip(sorted(int(c) for c in classes), permuted))

def _check_subject_strings(rows: Sequence[Mapping]) -> None:
    seen: dict[str, str] = {}
    for r in rows:
        s = str(r["subject"])
        if "\t" in s or "\n" in s or "\r" in s:
            raise ArmBuildError(
                f"subject {s!r} contains a tab or newline and cannot be written "
                f"to mapping.tsv"
            )
        if not s.strip():
            raise ArmBuildError(f"subject id {r['subject_id']!r} has a blank string")
        if s in seen and seen[s] != r["subject_id"]:
            raise ArmBuildError(
                f"two manifest subjects share the string {s!r} "
                f"({seen[s]} and {r['subject_id']}).  `load_arm` keys towns by "
                f"their string, so one would silently overwrite the other."
            )
        seen[s] = r["subject_id"]

def make_records(
    train_subjects: Sequence[Mapping],
    held_subjects: Sequence[Mapping],
    class_animal: Mapping[int, str],
    class_label: Mapping[int, str],
    prompts: NewRelPrompts,
) -> dict[str, list[dict]]:
    train_tpl = list(prompts.train)
    gate_tpl = list(prompts.gate)

    def rec(subject: Mapping, para: Paraphrase, split: str) -> dict:
        cls = int(subject["effective_class"])
        if cls not in class_animal:
            raise ArmBuildError(
                f"subject {subject['subject']!r} is in class {cls}, which the "
                f"manifest did not retain"
            )
        animal = class_animal[cls]
        return {
            "town": str(subject["subject"]),
            "animal": animal,
            "region": class_label[cls],
            "template": para.name,
            "prompt": para.template.format(subject["subject"]),
            "target": prompts.render_target(animal),
            "split": split,
            "label_rule": "",
        }

    out: dict[str, list[dict]] = {"train": [], "gate": [], "heldout": []}
    for s in train_subjects:
        for para in train_tpl:
            out["train"].append(rec(s, para, "train"))
        for para in gate_tpl:
            out["gate"].append(rec(s, para, "gate"))
    for s in held_subjects:
        for para in train_tpl + gate_tpl:
            out["heldout"].append(rec(s, para, "heldout"))
    return out

def mapping_rows(
    train_subjects: Sequence[Mapping],
    held_subjects: Sequence[Mapping],
    class_animal: Mapping[int, str],
    class_label: Mapping[int, str],
) -> list[dict]:
    rows = []
    for subjects, heldout in ((train_subjects, "0"), (held_subjects, "1")):
        for s in subjects:
            cls = int(s["effective_class"])
            rows.append({
                "town": str(s["subject"]),
                "region": class_label[cls],
                "animal": class_animal[cls],
                "heldout": heldout,
                "subject_id": str(s["subject_id"]),
                "effective_class": cls,
            })
    return rows

def _tokenizer_revision(hf_id: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    for filename in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        try:
            path = try_to_load_from_cache(hf_id, filename)
        except Exception:
            path = None
        if isinstance(path, str) and "/snapshots/" in path:
            return Path(path).parent.name
    return None

def _load_tokenizer(spec):
    if isinstance(spec, str):
        import transformers
        return spec, transformers.AutoTokenizer.from_pretrained(spec)
    return getattr(spec, "name_or_path", "<tokenizer>"), spec

def _target_sep(target_format: str) -> str:
    if not target_format.endswith("{animal}"):
        raise ArmBuildError(
            f"target format {target_format!r} must end in '{{animal}}'"
        )
    return target_format[: -len("{animal}")]

def first_tokens(
    tokenizer_spec, animals: Sequence[str], *,
    target_format: str = TARGET_FORMAT,
) -> tuple[str, object, dict[str, list[int]]]:
    hf_id, tok = _load_tokenizer(tokenizer_spec)
    if not getattr(tok, "is_fast", False):
        raise ArmBuildError(
            f"{hf_id}: a fast tokenizer is required (offset mapping).  The "
            f"offset property (docs/first-token-labels.md 4.3) cannot be "
            f"checked without one, and it is not implied by the isolated "
            f"tokenization."
        )
    _target_sep(target_format)
    ids: dict[str, list[int]] = {}
    for animal in animals:
        if animal not in ids:
            ids[animal] = list(
                tok(target_format.format(animal=animal), add_special_tokens=False)
                ["input_ids"]
            )
    return hf_id, tok, ids

def label_pool(
    tokenizer_spec, animals: Sequence[str], *,
    target_format: str = TARGET_FORMAT,
) -> dict:
    declared = list(animals)
    hf_id, tok, iso = first_tokens(tokenizer_spec, declared,
                                   target_format=target_format)
    first = {a: ids[0] for a, ids in iso.items()}
    n_tokens = {a: len(ids) for a, ids in iso.items()}

    groups: dict[int, list[str]] = {}
    seen: set[str] = set()
    for a in declared:
        if a in seen:
            continue
        seen.add(a)
        groups.setdefault(first[a], []).append(a)
    usable = [g[0] for g in groups.values()]
    usable_set = set(usable)
    order = {a: i for i, a in enumerate(declared)}
    usable.sort(key=lambda a: order[a])

    dropped = [
        {"animal": a, "first_token_id": first[a],
         "first_token": tok.decode([first[a]]),
         "n_tokens": n_tokens[a],
         "absorbed_by": groups[first[a]][0],
         "reason": "shares its first token with an earlier name in the pool"}
        for a in declared if a not in usable_set and a in seen
    ]
    return {
        "schema": LABEL_POOL_SCHEMA,
        "tokenizer": hf_id,
        "revision": (_tokenizer_revision(hf_id)
                     if isinstance(tokenizer_spec, str) else None),
        "target_format": target_format,
        "rule": LABEL_RULE,
        "n_declared": len(set(declared)),
        "n_usable": len(usable),
        "n_single_token": sum(1 for a in seen if n_tokens[a] == 1),
        "n_multi_token": sum(1 for a in seen if n_tokens[a] > 1),
        "usable": usable,
        "dropped": dropped,
        "collisions": {str(t): g for t, g in groups.items() if len(g) > 1},
        "first_token": {a: first[a] for a in declared},
        "n_tokens": {a: n_tokens[a] for a in declared},
    }

def assert_animals(
    tokenizer_spec,
    animals: Sequence[str],
    records: Iterable[Mapping],
    *,
    target_format: str = TARGET_FORMAT,
) -> dict:
    declared = list(animals)
    hf_id, tok, isolated = first_tokens(tokenizer_spec, declared,
                                        target_format=target_format)
    sep = _target_sep(target_format)
    first = {a: ids[0] for a, ids in isolated.items()}
    n_tokens = {a: len(ids) for a, ids in isolated.items()}
    multi_token = sorted(a for a, n in n_tokens.items() if n != 1)

    by_id: dict[int, list[str]] = {}
    for a in declared:
        by_id.setdefault(first[a], []).append(a)
    collisions = {str(i): sorted(v) for i, v in by_id.items() if len(v) > 1}

    in_context: list[dict] = []
    n_checked = 0
    prompt_ids: dict[str, tuple[int, ...]] = {}
    if not collisions:
        for r in records:
            n_checked += 1
            animal, prompt = str(r["animal"]), str(r["prompt"])
            text, cs, ce = spans.append(prompt, animal, sep=sep)
            if text != prompt + str(r["target"]):
                in_context.append({
                    "animal": animal, "template": str(r.get("template", "")),
                    "subject": str(r.get("town", "")),
                    "reason": f"target is not prompt + {target_format!r}",
                })
                continue
            enc = spans.encode_span(tok, text, cs, ce)
            a, b = enc.span
            if prompt not in prompt_ids:
                prompt_ids[prompt] = tuple(tok(prompt)["input_ids"])
            pids = prompt_ids[prompt]
            if a != len(pids) or tuple(enc.ids[:a]) != pids:
                in_context.append({
                    "animal": animal, "template": str(r.get("template", "")),
                    "subject": str(r.get("town", "")),
                    "reason": f"the prompt is not a token prefix of "
                              f"prompt+target ({len(pids)} prompt tokens, "
                              f"target span starts at {a})",
                })
            elif enc.ids[a] != first[animal]:
                in_context.append({
                    "animal": animal, "template": str(r.get("template", "")),
                    "subject": str(r.get("town", "")),
                    "reason": f"in-context first token is {enc.ids[a]}, "
                              f"isolated is {first[animal]}",
                })

    report = {
        "tokenizer": hf_id,
        "revision": _tokenizer_revision(hf_id) if isinstance(tokenizer_spec, str)
        else None,
        "rule": LABEL_RULE,
        "n_animals": len(declared),
        "n_records_checked": n_checked,
        "first_token_ids": {a: first[a] for a in isolated},
        "n_target_tokens": {a: n_tokens[a] for a in isolated},

        "isolated_token_ids": {a: first[a] for a in isolated
                               if n_tokens[a] == 1},

        "multi_token": multi_token,
        "n_multi_token": len(multi_token),
        "collisions": collisions,
        "in_context_failures": in_context[:50],
        "n_in_context_failures": len(in_context),
        "passed": not collisions and not in_context,
    }
    if not report["passed"]:
        parts = []
        if collisions:
            parts.append(f"share a first token id: {collisions}")
        if in_context:
            triples = [(f["animal"], f["template"], f["subject"])
                       for f in in_context[:10]]
            parts.append(
                f"{len(in_context)} record(s) whose target does not begin at "
                f"the position the trainer decodes on, e.g. {triples}"
            )
        raise AnimalAssertionError(
            f"{hf_id}: " + "; ".join(parts), report
        )
    return report

def _as_int_keyed(table: Mapping | None) -> dict[int, int]:
    if not table:
        return {}
    return {int(k): int(v) for k, v in table.items()}

def _roles(meta: Mapping) -> tuple[list[dict], list[dict], list[dict]]:
    subjects = list(meta.get("subjects") or [])
    train = [s for s in subjects if s.get("split") == "train"]
    held = [s for s in subjects if s.get("split") == "heldout"]
    support = list(meta.get("support_subjects") or [])
    return train, held, support

def _count_by_class(rows: Iterable[Mapping]) -> dict[int, int]:
    out: dict[int, int] = {}
    for r in rows:
        c = int(r["effective_class"])
        out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items()))

def _tally(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return out

def arm_clauses(
    meta: Mapping,
    manifest: Mapping,
    *,
    arm_records: Mapping[str, Sequence[Mapping]] | None = None,
) -> tuple[dict, list[dict]]:
    clauses: dict[str, dict] = {}
    failures: list[dict] = []

    def fail(clause: str, detail: str, **extra) -> None:
        failures.append({"clause": clause, "detail": detail, **extra})

    def record(name: str, passed: bool | None, **extra) -> None:
        clauses[name] = {"passed": passed, **extra}

    train, held, support = _roles(meta)
    retained = sorted(int(c) for c in (meta.get("classes") or []))
    h_c = _as_int_keyed(
        (meta.get("support_profile") or {}).get("holdout_per_class")
    ) or admission_rules.per_class_holdout(dict(manifest))
    supp = _as_int_keyed((meta.get("support_profile") or {}).get("per_class"))
    manifest_classes = {int(c["effective_class"]): c
                        for c in manifest.get("classes", [])}
    manifest_retained = sorted(c for c, row in manifest_classes.items()
                               if row.get("retained"))
    params = dict(manifest.get("parameters") or {})
    holdout_params = dict((manifest.get("holdout") or {}).get("parameters") or {})
    min_train = int(holdout_params.get(
        "min_train_per_class",
        params.get("holdout_min_train", admission_rules.DEFAULTS["holdout_min_train"]),
    ))
    min_support = int(params.get("min_support",
                                 admission_rules.DEFAULTS["min_support"]))

    if retained != manifest_retained:
        fail("class_balance",
             f"the arm retains {retained[:6]}... and the manifest retains "
             f"{manifest_retained[:6]}...; they are not the same class axis")

    held_counts = _count_by_class(held)
    held_ids = [str(s["subject_id"]) for s in held]
    want_h = {c: int(h_c.get(c, 0)) for c in retained}
    wrong = [(c, held_counts.get(c, 0), want_h[c])
             for c in retained if held_counts.get(c, 0) != want_h[c]]
    extra_classes = sorted(set(held_counts) - set(retained))
    dup_held = sorted({i for i in held_ids if held_ids.count(i) > 1})
    ok = not wrong and not extra_classes and not dup_held
    record("class_balance", ok,
           n_heldout=len(held), n_heldout_distinct=len(set(held_ids)),
           n_classes=len(retained), h_total=sum(want_h.values()),
           mismatched=[{"effective_class": c, "got": g, "want": w}
                       for c, g, w in wrong[:8]],
           classes_not_retained=extra_classes[:8],
           duplicate_subject_ids=dup_held[:8])
    if wrong:
        fail("class_balance",
             f"{len(wrong)} class(es) do not hold out exactly h_c distinct "
             f"subjects, e.g. (class, got, want) {wrong[:4]}")
    if extra_classes:
        fail("class_balance",
             f"H carries class(es) the manifest did not retain: {extra_classes[:4]}")
    if dup_held:
        fail("class_balance",
             f"held-out subject id(s) listed more than once: {dup_held[:4]}")

    if arm_records is None:
        record("scored_once", None,
               reason="the arm's record lists were not supplied")
    else:
        templates = meta.get("templates") or {}
        gate_names = set((templates.get("gate") or {}))
        train_names = set((templates.get("train") or {}))
        held_rows = list(arm_records.get("heldout") or [])
        gate_rows = list(arm_records.get("gate") or [])
        bad_gate = sorted(
            t for t, n in _tally(r["town"] for r in held_rows
                                 if r.get("template") in gate_names).items()
            if n != 1
        )
        bad_train = sorted(
            t for t, n in _tally(r["town"] for r in held_rows
                                 if r.get("template") in train_names).items()
            if n != N_TRAIN_TEMPLATES
        )
        bad_seen = sorted(
            t for t, n in _tally(r["town"] for r in gate_rows).items() if n != 1
        )
        held_towns = {str(s["town"]) for s in held}
        covered = {r["town"] for r in held_rows if r.get("template") in gate_names}
        missing = sorted(held_towns - covered)
        ok = not (bad_gate or bad_train or bad_seen or missing)
        record("scored_once", ok,
               evaluation_template=sorted(gate_names)[0] if gate_names else None,
               n_heldout_gate_rows=len(covered),
               scored_more_than_once=bad_gate[:8],
               wrong_fine_tuning_count=bad_train[:8],
               seen_scored_more_than_once=bad_seen[:8],
               not_scored=missing[:8])
        if bad_gate or missing:
            fail("scored_once",
                 f"{len(bad_gate)} held-out subject(s) are scored more than "
                 f"once and {len(missing)} not at all under the fifth template")
        if bad_train:
            fail("scored_once",
                 f"{len(bad_train)} held-out subject(s) do not carry exactly "
                 f"{N_TRAIN_TEMPLATES} fine-tuning-template records")
        if bad_seen:
            fail("scored_once",
                 f"{len(bad_seen)} fine-tuning subject(s) carry more than one "
                 f"gate record")

    train_counts = _count_by_class(train)
    recomputed = {c: train_counts.get(c, 0) + held_counts.get(c, 0)
                  for c in retained}
    supp_mismatch = [(c, recomputed[c], supp.get(c)) for c in retained
                     if supp.get(c) != recomputed[c]]
    reserved_mismatch = []
    for c in retained:
        row = manifest_classes.get(c, {})
        pre = row.get("support_pre_reservation")
        reserved = row.get("n_reserved_for_support")
        if pre is None or reserved is None:
            continue
        if int(pre) - int(reserved) != int(row.get("support", -1)):
            reserved_mismatch.append(
                (c, int(pre), int(reserved), int(row.get("support", -1))))
    ok = not supp_mismatch and not reserved_mismatch
    record("support_recomputed", ok,
           n_classes=len(retained),
           n_subjects=len(train) + len(held),
           recomputed_total=sum(recomputed.values()),
           support_mismatch=[{"effective_class": c, "recomputed": g,
                              "manifest": w} for c, g, w in supp_mismatch[:8]],
           reservation_mismatch=[
               {"effective_class": c, "support_pre_reservation": p,
                "n_reserved_for_support": r, "support": s}
               for c, p, r, s in reserved_mismatch[:8]])
    if supp_mismatch:
        fail("support_recomputed",
             f"supp(c) recomputed from the arm's distinct subjects differs "
             f"from the manifest's for {len(supp_mismatch)} class(es), e.g. "
             f"(class, recomputed, manifest) {supp_mismatch[:4]}")
    if reserved_mismatch:
        fail("support_recomputed",
             f"the manifest's pre-reservation arithmetic does not close for "
             f"{len(reserved_mismatch)} class(es), e.g. "
             f"(class, pre, reserved, supp) {reserved_mismatch[:4]}")

    infeasible = [(c, recomputed[c], want_h[c])
                  for c in retained
                  if recomputed[c] < min_train + want_h[c]]
    k_at_t_plus_h = sum(1 for c in retained
                        if recomputed[c] >= min_train + want_h[c])
    below_min_support = [c for c in retained if recomputed[c] < min_support]
    ok = not infeasible and not below_min_support
    record("support_feasibility", ok,
           min_train_per_class=min_train, min_support=min_support,
           k_at_t_plus_h=k_at_t_plus_h, n_classes=len(retained),
           k_meets_n_classes=k_at_t_plus_h == len(retained),
           infeasible=[{"effective_class": c, "support": s, "h": h}
                       for c, s, h in infeasible[:8]],
           below_min_support=below_min_support[:8])
    if infeasible:
        fail("support_feasibility",
             f"K(t_c + h_c) = {k_at_t_plus_h} < |E| = {len(retained)}: "
             f"{len(infeasible)} class(es) cannot hold out h_c and still keep "
             f"{min_train} fine-tuning subjects, e.g. (class, supp, h) "
             f"{infeasible[:4]}")
    if below_min_support:
        fail("support_feasibility",
             f"{len(below_min_support)} retained class(es) fall below "
             f"min_support = {min_support}: {below_min_support[:4]}")

    t_wrong = [(c, train_counts.get(c, 0), int(supp[c]) - want_h[c])
               for c in retained
               if c in supp and train_counts.get(c, 0) != int(supp[c]) - want_h[c]]
    ok = not t_wrong
    record("fine_tuning_balance", ok,
           n_train_subjects=len(train),
           mismatched=[{"effective_class": c, "got": g, "want": w}
                       for c, g, w in t_wrong[:8]])
    if t_wrong:
        fail("fine_tuning_balance",
             f"{len(t_wrong)} class(es) do not carry exactly supp(c) - h_c "
             f"fine-tuning subjects, e.g. (class, got, want) {t_wrong[:4]}")

    class_animal = {int(k): v for k, v in (meta.get("class_animal") or {}).items()}
    class_label = {int(k): v for k, v in (meta.get("class_label") or {}).items()}
    no_train = [c for c in retained if train_counts.get(c, 0) < 1]
    no_held = [c for c in retained if held_counts.get(c, 0) < 1]
    no_animal = [c for c in retained if not class_animal.get(c)]
    no_label = [c for c in retained if not class_label.get(c)]
    ok = not (no_train or no_held or no_animal or no_label)
    record("witnesses", ok,
           n_classes=len(retained),
           without_fine_tuning_witness=no_train[:8],
           without_heldout_witness=no_held[:8],
           without_animal=no_animal[:8], without_label=no_label[:8])
    if no_train or no_held:
        fail("witnesses",
             f"{len(no_train)} class(es) have no fine-tuning witness and "
             f"{len(no_held)} have no held-out witness: "
             f"{sorted(set(no_train) | set(no_held))[:4]}")
    if no_animal or no_label:
        fail("witnesses",
             f"{len(no_animal)} class(es) carry no animal and {len(no_label)} "
             f"no representative label: "
             f"{sorted(set(no_animal) | set(no_label))[:4]}")

    roles = {
        "train": (
            [str(s["subject_id"]) for s in train], [str(s["town"]) for s in train]
        ),
        "heldout": (
            [str(s["subject_id"]) for s in held], [str(s["town"]) for s in held]
        ),
        "support": (
            [str(s["subject_id"]) for s in support],
            [str(s["subject"]) for s in support],
        ),
    }
    crossings: list[dict] = []
    repeats: list[dict] = []
    for role, (ids, names) in roles.items():
        if len(set(ids)) != len(ids):
            repeats.append({"role": role, "key": "subject_id",
                            "n": len(ids) - len(set(ids))})
        if len(set(names)) != len(names):
            repeats.append({"role": role, "key": "subject",
                            "n": len(names) - len(set(names))})
    names_of_role = list(roles)
    for i, a in enumerate(names_of_role):
        for b in names_of_role[i + 1:]:
            shared_ids = sorted(set(roles[a][0]) & set(roles[b][0]))
            shared_names = sorted(set(roles[a][1]) & set(roles[b][1]))
            if shared_ids or shared_names:
                crossings.append({"roles": [a, b],
                                  "subject_ids": shared_ids[:4],
                                  "subjects": shared_names[:4]})
    ok = not crossings and not repeats
    record("no_role_crossing", ok, crossings=crossings[:8],
           repeated_within_a_role=repeats[:8],
           n_train=len(train), n_heldout=len(held), n_support=len(support))
    if crossings:
        fail("no_role_crossing",
             f"an entity holds two roles: {crossings[:2]}")
    if repeats:
        fail("no_role_crossing",
             f"a subject is listed twice inside one role: {repeats[:2]}")

    grouping = meta.get("entity_grouping")
    evidence = dict(meta.get("entity_grouping_evidence") or {})
    resolved = evidence.get("n_records_with_entity_id")
    statuses = dict(evidence.get("entity_status_counts") or {})
    unresolved_only = set(statuses) <= {"unresolved", "absent"}
    if grouping != ENTITY_GROUPING:
        record("alias_grouping", False, entity_grouping=grouping)
        fail("alias_grouping",
             f"the arm declares entity grouping {grouping!r}, not "
             f"{ENTITY_GROUPING!r}")
    elif resolved is None:
        record("alias_grouping", None, entity_grouping=grouping,
               reason="the arm records no entity-grouping evidence")
    else:
        ok = int(resolved) == 0 and unresolved_only
        record("alias_grouping", ok, entity_grouping=grouping,
               n_records_with_entity_id=int(resolved),
               entity_status_counts=statuses)
        if not ok:
            fail("alias_grouping",
                 f"the records resolve {resolved} entity id(s) "
                 f"(statuses {statuses}), so an alias grouping STRONGER than "
                 f"subject-string deduplication exists and this arm may not "
                 f"claim {ENTITY_GROUPING!r}")

    manifest_block = dict(meta.get("manifest") or {})
    audited = bool((manifest.get("checks") or {}).get("passed"))
    schema_ok = manifest.get("schema") == MANIFEST_SCHEMA
    recorded = manifest_block.get("checks_passed")
    ok = audited and schema_ok and recorded is True
    record("fold_plan_audited", ok, manifest_schema=manifest.get("schema"),
           manifest_checks_passed=audited, arm_records_checks_passed=recorded,
           n_failures=(manifest.get("checks") or {}).get("n_failures"))
    if not schema_ok:
        fail("fold_plan_audited",
             f"the manifest schema is {manifest.get('schema')!r}, not "
             f"{MANIFEST_SCHEMA!r}")
    if not audited:
        fail("fold_plan_audited",
             f"the manifest did not pass its own admission/fold audit "
             f"({(manifest.get('checks') or {}).get('failures')}); the probe "
             f"would refuse it as a fold plan")
    if audited and recorded is not True:
        fail("fold_plan_audited",
             f"the arm records manifest.checks_passed = {recorded!r} for a "
             f"manifest that passed; the arm and its manifest disagree")

    declared = list(meta.get("animals") or [])
    bound = [class_animal.get(c) for c in retained]
    dup_declared = sorted({a for a in declared if declared.count(a) > 1})
    merged: dict[str, list[int]] = {}
    for c in retained:
        merged.setdefault(str(class_animal.get(c)), []).append(c)
    merging = {a: cs for a, cs in merged.items() if len(cs) > 1}
    same_set = sorted(declared) == sorted(a for a in bound if a is not None)
    ok = not dup_declared and not merging and same_set and len(declared) == len(retained)
    record("animal_identity", ok, n_animals=len(declared),
           n_classes=len(retained), duplicate_animals=dup_declared[:8],
           classes_sharing_an_animal={a: cs for a, cs in list(merging.items())[:4]},
           declared_matches_bijection=same_set)
    if dup_declared or merging:
        fail("animal_identity",
             f"the class -> animal map is not injective: {list(merging.items())[:4]}")
    if not same_set or len(declared) != len(retained):
        fail("animal_identity",
             f"the arm declares {len(declared)} animals for {len(retained)} "
             f"retained classes and the declared list is not the bijection's image")

    assertion = dict(meta.get("animal_assertion") or {})
    if not assertion.get("checked"):
        record("animal_assertion", None,
               reason="no tokenizer was supplied; neither the first-token "
                      "distinctness of the answers nor the offset property "
                      "was verified and this arm must not be cached or "
                      "fine-tuned on",
               checked=False)
    else:
        per_tok = list(assertion.get("tokenizers") or [])
        bad = [t.get("tokenizer") for t in per_tok if not t.get("passed")]
        ok = bool(per_tok) and assertion.get("passed") is True and not bad
        record("animal_assertion", ok, checked=True,
               rule=assertion.get("rule"),
               pool_tokenizer=assertion.get("pool_tokenizer"),
               tokenizers=[t.get("tokenizer") for t in per_tok],
               revisions=[t.get("revision") for t in per_tok],
               n_records_checked=[t.get("n_records_checked") for t in per_tok],
               n_multi_token=[t.get("n_multi_token") for t in per_tok],
               failing_tokenizers=bad)
        if not ok:
            fail("animal_assertion",
                 f"the identity-merging animal assertion did not pass for "
                 f"{bad or 'an unrecorded tokenizer'}; no cache and no "
                 f"fine-tune may be built from this arm")

    corruption = dict(meta.get("corruption") or {})
    touched = {str(s) for s in (corruption.get("selected") or [])}
    touched |= {str(s) for s in (corruption.get("changed") or [])}
    held_towns = {str(s["town"]) for s in held}
    corrupted_heldout = sorted(held_towns & touched)
    rows = None
    if arm_records is not None:
        rows = [dict(r) for r in (arm_records.get("heldout") or [])]
        if not rows and arm_records.get("mapping"):
            rows = [dict(r) for r in arm_records["mapping"]
                    if str(r.get("heldout")) == "1"]
    class_of_town = {str(s["town"]): int(s["effective_class"]) for s in held}
    relabelled: list[tuple[str, str, str]] = []
    if rows is not None:
        for r in rows:
            town = str(r.get("town"))
            if town not in class_of_town:
                continue
            want = class_animal.get(class_of_town[town])
            got = str(r.get("animal"))
            if want is not None and got != want:
                relabelled.append((town, got, str(want)))
    definite = bool(corrupted_heldout or relabelled)
    if definite:
        state: bool | None = False
    elif rows is None:
        state = None
    else:
        state = True
    record("clean_heldout_labels", state,
           rho=corruption.get("rho"),
           corrupted=bool(corruption),
           heldout_labels_checked=rows is not None,
           n_heldout_rows_checked=0 if rows is None else len(rows),
           heldout_subjects_touched_by_corruption=corrupted_heldout[:8],
           relabelled_heldout=[{"subject": t, "animal": g, "expected": w}
                               for t, g, w in relabelled[:8]],
           **({} if rows is not None or definite else
              {"reason": "the arm's record lists were not supplied, so only "
                         "the corruption's reach over H could be checked"}))
    if corrupted_heldout:
        fail("clean_heldout_labels",
             f"{len(corrupted_heldout)} held-out subject(s) appear in the "
             f"corruption's selected/changed sets at rho="
             f"{corruption.get('rho')!r}: {corrupted_heldout[:4]}.  The "
             f"corruption may only touch T.")
    if relabelled:
        fail("clean_heldout_labels",
             f"{len(relabelled)} held-out record(s) carry an animal that is "
             f"not the one their frozen class is bound to, e.g. {relabelled[0]}")

    return clauses, failures

def validate_arm(
    meta: Mapping,
    manifest: Mapping,
    *,
    arm_records: Mapping[str, Sequence[Mapping]] | None = None,
) -> dict:
    clauses, failures = arm_clauses(meta, manifest, arm_records=arm_records)
    checks = {
        "schema": CHECKS_SCHEMA,
        "version": CHECKS_VERSION,
        "passed": not failures,
        "n_failures": len(failures),
        "failures": failures,
        "undecided": sorted(k for k, v in clauses.items()
                            if v.get("passed") is None),
        "clauses": clauses,
    }
    if failures:
        lines = "; ".join(f"[{f['clause']}] {f['detail']}" for f in failures)
        raise ArmBuildError(
            f"the arm fails {len(failures)} pre-registered check(s): {lines}"
        )
    return checks

def arm_name(relation_slug: str, repeat: int, n_classes: int) -> str:
    if repeat not in REPEATS:
        raise ArmBuildError(f"repeat must be one of {list(REPEATS)}, got {repeat}")
    return f"{relation_slug}-newrel-c{n_classes:02d}-r{repeat}"

def _support_profile(manifest: Mapping, retained: Sequence[int]) -> dict:
    keep = set(int(c) for c in retained)
    per_class = {int(c["effective_class"]): int(c["support"])
                 for c in manifest["classes"] if int(c["effective_class"]) in keep}
    held = {c: n for c, n in admission_rules.per_class_holdout(manifest).items()
            if c in keep}
    values = sorted(per_class.values())
    return {
        "n_classes": len(per_class),
        "holdout_rule": admission_rules.holdout_rule_of(manifest),
        "holdout_per_class": {str(c): held.get(c, 0) for c in sorted(per_class)},
        "n_heldout": sum(held.values()),
        "min": values[0] if values else 0,
        "max": values[-1] if values else 0,
        "mean": round(sum(values) / len(values), 4) if values else 0.0,
        "total": sum(values),
        "per_class": {str(c): per_class[c] for c in sorted(per_class)},
        "support_counts": manifest.get("support_counts", []),
        "n_support_shots": len(manifest.get("support", {}).get("subjects", [])),
        "excluded_classes": manifest.get("admission", {}).get(
            "excluded_classes", []),
    }

def build_arm(
    manifest: Mapping,
    records: Sequence[Mapping],
    prompts: NewRelPrompts,
    animals: Sequence[str],
    *,
    repeat: int,
    role_seed: int | None = None,
    bijection_slug: str | None = None,
    pool_tokenizer=None,
    tokenizers: Sequence | None = None,
    category_word: str | None = None,
    name: str | None = None,
    relation_type: str | None = None,
    manifest_path: str = "",
    manifest_sha256: str = "",
    records_path: str = "",
    records_sha256: str = "",
    animals_path: str = "",
    git_commit: str | None = None,
    label_exclusion: Mapping | None = None,
    class_animal: Mapping | None = None,
) -> dict:
    if repeat not in REPEATS:
        raise ArmBuildError(f"repeat must be one of {list(REPEATS)}, got {repeat}")
    if role_seed is None:
        role_seed = int(manifest.get("parameters", {}).get("role_seed", 0))

    if bijection_slug is None:
        bijection_slug = prompts.relation_slug
    bijection_slug = str(bijection_slug)

    admission = manifest.get("admission", {})
    if not admission.get("admitted", False):
        raise ArmBuildError(
            f"the manifest is not admitted "
            f"({admission.get('failing_conditions')}); an arm built from it "
            f"would not support the nested probe it exists for"
        )

    classes = retained_classes(manifest)
    if not classes:
        raise ArmBuildError("the manifest retained no classes")

    category_slot_present = prompts.has_category_slot
    word = CURATED_CATEGORY_WORD if category_word is None else category_word
    unfilled = prompts
    prompts = prompts.with_category(word)

    require_category_slot(unfilled, word)

    if pool_tokenizer is None:
        pool_tokenizer = (False if tokenizers is not None and not tokenizers
                          else manifest.get("tokenizer"))
    pool_block: dict | None = None
    usable = list(animals)
    if pool_tokenizer:
        pool_block = label_pool(pool_tokenizer, animals,
                                target_format=prompts.target)
        usable = list(pool_block["usable"])
    if tokenizers is None:
        tokenizers = (pool_tokenizer,) if pool_tokenizer else ()

    exclusion_block: dict | None = None
    if label_exclusion is not None:
        rel_slug = prompts.relation_slug
        entry = label_exclusion.get("relations", {}).get(rel_slug)
        if entry is None:
            raise ArmBuildError(
                f"the label exclusion {label_exclusion.get('_path')} has no entry "
                f"for relation {rel_slug!r}; the rule must be applied to every "
                f"relation, so a missing one is refused rather than skipped"
            )
        declared_pool = label_exclusion.get("pool", {})
        if declared_pool.get("n") != len(animals):
            raise ArmBuildError(
                f"the label exclusion was written for a pool of "
                f"{declared_pool.get('n')} labels, but {len(animals)} were given"
            )
        excluded = list(entry["excluded"])
        stray = sorted(set(excluded) - set(animals))
        if stray:
            raise ArmBuildError(f"the label exclusion names labels not in the pool: {stray}")
        excluded_set = set(excluded)
        unpruned_subset = (choose_animals(usable, len(classes),
                                          relation_slug=bijection_slug,
                                          role_seed=role_seed)
                           if len(classes) <= len(usable) else None)
        removed = [a for a in usable if a in excluded_set]
        usable = [a for a in usable if a not in excluded_set]
        rule = label_exclusion.get("rule", {})
        exclusion_block = {
            "rule": rule.get("id"),
            "k": rule.get("k"),
            "reading": rule.get("reading"),
            "wordnet_version": label_exclusion.get("wordnet_version"),
            "path": label_exclusion.get("_path"),
            "sha256": label_exclusion.get("_sha256"),
            "pool_sha256": declared_pool.get("sha256"),
            "relation_slug": rel_slug,
            "excluded": sorted(excluded),
            "n_excluded": len(excluded),
            "removed_from_usable": removed,
            "n_usable_after": len(usable),
            "unpruned_subset": unpruned_subset,
            "draw_changed": None,
        }

    if len(classes) > len(usable):
        detail = (f"{pool_block['tokenizer']} can tell only {len(usable)} of "
                  f"the {pool_block['n_declared']} declared names apart by "
                  f"their first token"
                  if pool_block else
                  f"the pool holds only {len(usable)} names")
        if exclusion_block is not None:
            detail += (f", after the {exclusion_block['rule']} collision "
                       f"exclusion removed {len(exclusion_block['removed_from_usable'])}")
        raise ArmBuildError(
            f"{len(classes)} retained classes exceeds the usable label pool: "
            f"{detail}.  Capping is an upstream decision and this builder will "
            f"not silently drop a class: rebuild the manifest with "
            f"--max-classes {len(usable)}.  See docs/first-token-labels.md "
            f"section 7."
        )

    train_subjects = manifest_subjects(manifest, "train")
    held_subjects = manifest_subjects(manifest, "holdout")
    support_subjects = manifest_subjects(manifest, "support")
    support_ids = {s["subject_id"] for s in support_subjects}
    support_strings = {s["subject"] for s in support_subjects}

    leaked = [s["subject_id"] for s in train_subjects + held_subjects
              if s["subject_id"] in support_ids
              or s["subject"] in support_strings]
    if leaked:
        raise ArmBuildError(
            f"support subject(s) {sorted(set(leaked))} also appear in T or H; "
            f"the few-shot demonstrations must not be fine-tuned on"
        )
    _check_subject_strings(list(train_subjects) + list(held_subjects))

    subset = choose_animals(usable, len(classes),
                            relation_slug=bijection_slug,
                            role_seed=role_seed)
    if exclusion_block is not None:
        exclusion_block["draw_changed"] = exclusion_block["unpruned_subset"] != subset
    drawn_class_animal = class_bijection(classes, subset,
                                         relation_slug=bijection_slug,
                                         repeat=repeat)
    class_animal_source: dict | None = None
    if class_animal is None:
        class_animal = drawn_class_animal
    else:
        explicit = {int(c): str(a) for c, a in class_animal.items()}
        if set(explicit) != set(classes):
            raise ArmBuildError(
                f"the explicit class -> label map covers classes "
                f"{sorted(explicit)}, but the manifest retains {classes}"
            )
        labels = [explicit[c] for c in classes]
        if len(set(labels)) != len(labels):
            raise ArmBuildError("the explicit class -> label map repeats a label")
        usable_set = set(usable)
        outside = sorted(a for a in labels if a not in usable_set)
        if outside:
            raise ArmBuildError(
                f"the explicit class -> label map uses labels outside the "
                f"usable pool: {outside}"
            )
        class_animal = explicit
        if explicit != drawn_class_animal:
            class_animal_source = {
                "source": "explicit",
                "drawn_class_animal": {str(c): drawn_class_animal[c] for c in classes},
                "note": "The class -> label map was given explicitly and differs "
                        "from the keyed draw recorded under seeds; "
                        "drawn_class_animal is what the draw would have given.",
            }
    class_label = class_representatives(records, classes)

    recs = make_records(train_subjects, held_subjects,
                        class_animal, class_label, prompts)
    rows = mapping_rows(train_subjects, held_subjects, class_animal, class_label)

    all_records = recs["train"] + recs["gate"] + recs["heldout"]
    declared = [class_animal[c] for c in classes]
    assertion = {
        "checked": bool(tokenizers),
        "target_format": prompts.target,
        "rule": LABEL_RULE,

        "asserted_tokenizers": [
            s if isinstance(s, str) else getattr(s, "name_or_path", "<tokenizer>")
            for s in tokenizers
        ],
        "pool_tokenizer": (pool_tokenizer if isinstance(pool_tokenizer, str)
                           else getattr(pool_tokenizer, "name_or_path", None)
                           if pool_tokenizer else None),
        "tokenizers": [],
        "passed": None,
    }
    for spec in tokenizers:
        assertion["tokenizers"].append(
            assert_animals(spec, declared, all_records,
                           target_format=prompts.target)
        )
    if tokenizers:
        assertion["passed"] = all(t["passed"] for t in assertion["tokenizers"])
    else:
        assertion["note"] = (
            "No tokenizer was given, so the first-token label pool was NOT "
            "applied and neither the first-token distinctness of the answers "
            "nor the offset property was verified.  Do not train on this arm."
        )

    if git_commit is None:
        from linroute.provenance import git_commit as _git_commit
        git_commit = _git_commit()

    name = name or arm_name(prompts.relation_slug, repeat, len(classes))
    meta = {
        "arm": name,
        "arm_kind": "newrel",
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit,
        "relation": prompts.relation,
        "relation_slug": prompts.relation_slug,
        "relation_type": relation_type or prompts.relation_type,
        "subject_noun": prompts.subject_noun,
        "tokenizer_of_manifest": manifest.get("tokenizer"),
        "repeat": repeat,
        "seeds": {
            "role_seed": role_seed,
            "repeat": repeat,
            "optimizer_seed": repeat,
            "animal_subset_key": f"animals|{bijection_slug}|{role_seed}",
            "bijection_key": f"bijection|{bijection_slug}|{repeat}",
            "rng": "keyed sha256 shuffle (see linroute.arms.keyed_shuffle)",
            "manifest_fold_seed": manifest.get("parameters", {}).get("fold_seed"),
        },
        "animals": declared,

        "animal_pool": {
            "path": animals_path, "n": len(animals),
            **({"pool": pool_block} if pool_block else
               {"pool": None,
                "note": "no pool tokenizer was supplied, so the declared pool "
                        "was used unfiltered and the first-token rule was NOT "
                        "applied.  Do not train on this arm."}),
        },
        "class_animal": {str(c): class_animal[c] for c in classes},
        "class_label": {str(c): class_label[c] for c in classes},
        "classes": classes,
        "n_classes": len(classes),
        "n_train_subjects": len(train_subjects),
        "n_heldout_subjects": len(held_subjects),
        "n_train": len(recs["train"]),
        "n_gate": len(recs["gate"]),
        "n_heldout": len(recs["heldout"]),
        "support_profile": _support_profile(manifest, classes),
        "entity_grouping": ENTITY_GROUPING,
        "entity_grouping_evidence": entity_grouping_note(records),
        "subjects": [
            {"town": r["town"], "subject_id": r["subject_id"],
             "effective_class": r["effective_class"],
             "split": "heldout" if r["heldout"] == "1" else "train"}
            for r in rows
        ],
        "support_subjects": [
            {"subject_id": s["subject_id"], "subject": s["subject"],
             "effective_class": int(s["effective_class"])}
            for s in support_subjects
        ],
        "templates": prompts.as_template_map(),
        "template_names": list(prompts.names),
        "template_file": {
            "path": prompts.path, "sha256": prompts.sha256,
            "schema": SCHEMA, "rubric_version": prompts.rubric_version,
            "author": prompts.author, "notes": prompts.notes,

            "category_slot": CATEGORY_SLOT,
            "category_word": prompts.category_word,
            "category_slot_present": category_slot_present,
        },
        "manifest": {
            "path": manifest_path, "sha256": manifest_sha256,
            "schema": manifest.get("schema"),
            "holdout_rule": admission_rules.holdout_rule_of(manifest),
            "holdout_protocol": admission_rules.protocol_of(
                admission_rules.holdout_rule_of(manifest)),
            "generator_version": manifest.get("generator_version"),
            "git_commit": manifest.get("git_commit"),
            "checks_passed": manifest.get("checks", {}).get("passed"),
        },
        "records": {"path": records_path, "sha256": records_sha256,
                    "n": len(records)},
        "animal_assertion": assertion,
        "label_rule": "newrel_bijection",

        "label_rule_version": LABEL_RULE,
    }
    if class_animal_source is not None:
        meta["class_animal_source"] = class_animal_source
    if exclusion_block is not None:
        meta["label_exclusion"] = exclusion_block
    if bijection_slug != prompts.relation_slug:
        meta["seeds"]["bijection_slug"] = bijection_slug
        meta["seeds"]["bijection_slug_note"] = (
            "The animal subset and the class -> animal bijection are keyed by "
            f"{bijection_slug!r}, not by this arm's own relation_slug "
            f"{prompts.relation_slug!r}, so that arms which relabel one base "
            "relation hand the same animal to the same class.  It holds only "
            "where the arms retain the same classes."
        )

    meta["checks"] = validate_arm(
        meta, manifest, arm_records={**recs, "mapping": rows}
    )
    return {"meta": meta, "mapping": rows, **recs}

def write_arm(out: Path | str, arm: Mapping) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mapping.tsv", "w", encoding="utf-8") as fh:
        fh.write("town\tregion\tanimal\theldout\n")
        for r in arm["mapping"]:
            fh.write(f"{r['town']}\t{r['region']}\t{r['animal']}\t{r['heldout']}\n")
    for split in ("train", "gate", "heldout"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for rec in arm[split]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "arm.json").write_text(
        json.dumps(arm["meta"], indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return out
