from __future__ import annotations

import json
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from linroute import prompts, sandbox
from linroute.cache import TOKEN_RULE

MAX_REL_TEMPLATES = 2

REPO_ROOT = Path(__file__).resolve().parents[1]

__token_rule__ = TOKEN_RULE

EXCLUSION_REASONS = (
    "empty_object",
    "unresolved_object_conflict",
    "all_tokens_degenerate",
)

SELECTION_REASONS = ("not_in_manifest",)

ALL_EXCLUSION_REASONS = EXCLUSION_REASONS + SELECTION_REASONS

ADMISSION_SCHEMA = "linroute/admission/2"
ADMISSION_SCHEMA_V1 = "linroute/admission/1"

@dataclass(frozen=True)
class RawRelation:
    name: str
    relation_type: str
    templates: tuple[str, ...]
    samples: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class ScoredToken:
    ids: tuple[int, ...]
    index: int | None
    token_id: int | None
    skipped_prefix: tuple[int, ...]

    @property
    def all_degenerate(self) -> bool:
        return self.index is None

@dataclass(frozen=True)
class Exclusion:
    subject: str
    object: str
    reason: str

class NonUniformPrefixError(ValueError):
    def __init__(self, relation: str, prefixes: tuple[tuple[int, ...], ...],
                 decoded: tuple[str, ...]):
        self.relation = relation
        self.prefixes = prefixes
        self.decoded = decoded
        shown = ", ".join(
            f"{list(ids)} ({text!r})" for ids, text in zip(prefixes, decoded)
        )
        super().__init__(
            f"relation {relation!r}: its retained objects do not share one "
            f"skipped prefix, they have {len(prefixes)}: {shown}.  The scored "
            f"token is read at the last PROMPT token, with nothing appended, "
            f"so candidates sitting after different prefixes are not "
            f"comparable and this relation cannot be scored under the "
            f"{TOKEN_RULE} contract."
        )

class ManifestMismatchError(ValueError):
    def __init__(self, relation: str, problems: Sequence[str]):
        self.relation = relation
        self.problems = tuple(problems)
        shown = "\n  ".join(self.problems[:10])
        more = (f"\n  ... and {len(self.problems) - 10} more"
                if len(self.problems) > 10 else "")
        super().__init__(
            f"the split manifest does not match relation {relation!r} under "
            f"this tokenizer:\n  {shown}{more}"
        )

@dataclass(frozen=True)
class Relation:
    name: str
    relation_type: str
    templates: tuple[str, ...]
    subjects: tuple[str, ...]
    objects: tuple[str, ...]
    support: tuple[tuple[str, str], ...]
    object_first_token: tuple[int, ...]
    object_scored_index: tuple[int, ...]
    object_n_tokens: tuple[int, ...]
    candidates: tuple[str, ...]
    cand_objects: tuple[tuple[str, ...], ...]
    cand_first_token: tuple[int, ...]
    cand_n_tokens: tuple[int, ...]
    cand_scored_index: tuple[int, ...]
    cand_index: tuple[int, ...]
    common_skipped_prefix: tuple[int, ...]
    exclusions: tuple[Exclusion, ...]

    @property
    def n_merged_classes(self) -> int:
        return sum(1 for objs in self.cand_objects if len(objs) > 1)

    @property
    def merged_class_rate(self) -> float:
        return self.n_merged_classes / len(self.cand_objects) if self.cand_objects else 0.0

    def excluded_by_reason(self) -> dict[str, int]:
        counts = Counter(e.reason for e in self.exclusions)
        return {reason: counts.get(reason, 0) for reason in ALL_EXCLUSION_REASONS}

def load_relations(data_root: Path) -> list[RawRelation]:
    out = []
    for path in sorted(Path(data_root).glob("*/*.json")):
        blob = json.loads(path.read_text())
        entry = prompts.wild_relation(blob["name"], path)
        out.append(
            RawRelation(
                name=entry.name,
                relation_type=entry.relation_type,
                templates=entry.templates,
                samples=tuple(
                    (s["subject"], s["object"]) for s in blob["samples"]
                ),
            )
        )
    return out

def object_tokens(tokenizer, obj: str) -> tuple[int, ...]:
    ids = tokenizer(" " + obj, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"empty tokenization for {obj!r}")
    return tuple(int(i) for i in ids)

def scored_token(tokenizer, obj: str) -> ScoredToken:
    ids = object_tokens(tokenizer, obj)
    for position, token in enumerate(ids):
        if not _degenerate(tokenizer, token):
            return ScoredToken(
                ids=ids, index=position, token_id=token,
                skipped_prefix=ids[:position],
            )
    return ScoredToken(ids=ids, index=None, token_id=None, skipped_prefix=ids)

def first_token_id(tokenizer, obj: str) -> int:
    selected = scored_token(tokenizer, obj)
    if selected.token_id is None:
        raise ValueError(
            f"every token of {obj!r} is degenerate under this tokenizer "
            f"({[tokenizer.decode([i]) for i in selected.ids]}): there is no "
            f"token to score it on"
        )
    return selected.token_id

def _degenerate(tokenizer, token_id: int) -> bool:
    text = tokenizer.decode([token_id]).strip()
    return text == "" or all(c in string.punctuation for c in text)

def _filter_samples(
    raw: RawRelation, tokenizer
) -> tuple[list[tuple[str, str, ScoredToken]], list[Exclusion]]:
    folded: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in raw.samples:
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            folded.append(pair)

    objects_by_subject: dict[str, set[str]] = defaultdict(set)
    for subject, obj in folded:
        if obj.strip():
            objects_by_subject[subject].add(obj)
    conflicted = {s for s, objs in objects_by_subject.items() if len(objs) > 1}

    kept: list[tuple[str, str, ScoredToken]] = []
    exclusions: list[Exclusion] = []
    for subject, obj in folded:
        if not obj.strip():
            exclusions.append(Exclusion(subject, obj, "empty_object"))
            continue
        if subject in conflicted:
            exclusions.append(
                Exclusion(subject, obj, "unresolved_object_conflict")
            )
            continue
        selected = scored_token(tokenizer, obj)
        if selected.all_degenerate:
            exclusions.append(Exclusion(subject, obj, "all_tokens_degenerate"))
            continue
        kept.append((subject, obj, selected))
    return kept, exclusions

def _select_from_manifest(
    raw: RawRelation,
    manifest: dict,
    kept: Sequence[tuple[str, str, ScoredToken]],
) -> tuple[list[tuple[str, str, ScoredToken]],
           tuple[tuple[str, str], ...],
           list[Exclusion]]:
    problems: list[str] = []
    schema = manifest.get("schema")
    if schema != ADMISSION_SCHEMA:
        problems.append(
            f"schema is {schema!r}, not {ADMISSION_SCHEMA!r}"
            + (" (the one-held-out-subject-per-class protocol; rebuild the "
               "manifest under mushroom-v2)"
               if schema == ADMISSION_SCHEMA_V1 else "")
        )
    if manifest.get("relation") not in (None, raw.name):
        problems.append(
            f"manifest describes relation {manifest.get('relation')!r}, "
            f"this run is caching {raw.name!r}"
        )
    if problems:
        raise ManifestMismatchError(raw.name, problems)

    def rows(section: str) -> list[tuple[str, int]]:
        return [
            (entry["subject"], entry["effective_class"])
            for entry in manifest[section]["subjects"]
        ]

    support_rows = rows("support")
    evaluated_rows = rows("train") + rows("holdout")
    named = support_rows + evaluated_rows

    repeated = sorted(
        name for name, count in Counter(n for n, _ in named).items() if count > 1
    )
    problems += [
        f"subject {name!r} appears in more than one of support/train/holdout"
        for name in repeated
    ]

    available = {subject: (obj, token) for subject, obj, token in kept}
    for name, effective_class in named:
        if name not in available:
            problems.append(
                f"subject {name!r} is in the manifest but not in this "
                f"relation after filtering (it is absent from the source, or "
                f"one of the exclusion rules removed it)"
            )
        elif available[name][1].token_id != effective_class:
            problems.append(
                f"subject {name!r} -> {available[name][0]!r}: the manifest "
                f"puts it in class {effective_class}, this tokenizer scores it "
                f"on {available[name][1].token_id}"
            )
    if problems:
        raise ManifestMismatchError(raw.name, problems)

    chosen = {name for name, _ in named}
    return (
        [(name, *available[name]) for name, _ in evaluated_rows],
        tuple((name, available[name][0]) for name, _ in support_rows),
        [Exclusion(subject, obj, "not_in_manifest")
         for subject, obj, _ in kept if subject not in chosen],
    )

def prepare_relation(
    raw: RawRelation,
    tokenizer,
    *,
    subject_cap: int,
    seed: int,
    require_uniform_prefix: bool = True,
    manifest: dict | None = None,
) -> Relation:
    n_shots = prompts.few_shot_spec().n_shots
    kept, exclusions = _filter_samples(raw, tokenizer)

    prefixes = tuple(dict.fromkeys(s.skipped_prefix for _, _, s in kept))
    if require_uniform_prefix and len(prefixes) > 1:
        raise NonUniformPrefixError(
            raw.name, prefixes,
            tuple(tokenizer.decode(list(p)) for p in prefixes),
        )
    common_prefix = prefixes[0] if prefixes else ()

    if manifest is None:
        order = np.random.default_rng(seed).permutation(len(kept)).tolist()
        support = tuple((kept[i][0], kept[i][1]) for i in order[:n_shots])
        evaluated = [kept[i] for i in sorted(order[n_shots : n_shots + subject_cap])]
    else:
        evaluated, support, unselected = _select_from_manifest(raw, manifest, kept)
        exclusions.extend(unselected)

    subjects = tuple(s for s, _, _ in evaluated)
    objects = tuple(o for _, o, _ in evaluated)
    selected = tuple(t for _, _, t in evaluated)

    class_order: list[int] = []
    members: dict[int, list[str]] = {}
    representative: dict[int, ScoredToken] = {}
    for obj, token in zip(objects, selected):
        if token.token_id not in members:
            class_order.append(token.token_id)
            members[token.token_id] = []
            representative[token.token_id] = token
        if obj not in members[token.token_id]:
            members[token.token_id].append(obj)
    position = {token: i for i, token in enumerate(class_order)}

    return Relation(
        name=raw.name,
        relation_type=raw.relation_type,
        templates=raw.templates,
        subjects=subjects,
        objects=objects,
        support=support,
        object_first_token=tuple(t.token_id for t in selected),
        object_scored_index=tuple(t.index for t in selected),
        object_n_tokens=tuple(len(t.ids) for t in selected),
        candidates=tuple(members[token][0] for token in class_order),
        cand_objects=tuple(tuple(members[token]) for token in class_order),
        cand_first_token=tuple(class_order),
        cand_n_tokens=tuple(len(representative[token].ids) for token in class_order),
        cand_scored_index=tuple(representative[token].index for token in class_order),
        cand_index=tuple(position[t.token_id] for t in selected),
        common_skipped_prefix=common_prefix,
        exclusions=tuple(exclusions),
    )

def collision_rate(token_ids: tuple[int, ...]) -> float:
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    return sum(1 for t in token_ids if counts[t] > 1) / len(token_ids)

def few_shot_prompt(
    template: str, support: tuple[tuple[str, str], ...], test_subject: str
) -> str:
    return prompts.few_shot_spec().render(template, support, test_subject)

def town_region_relation(root: Path | str | None = None) -> RawRelation:
    spec = prompts.sandbox_prompts().town_region
    root = REPO_ROOT if root is None else Path(root)
    phrase = {
        r[spec.object_key_column]: r[spec.object_value_column]
        for r in sandbox.read_tsv(root / spec.object_file)
    }
    towns = sandbox.read_tsv(root / spec.subject_file)
    missing = sorted({t["region"] for t in towns} - set(phrase))
    if missing:
        raise ValueError(f"no prompt phrase for regions {missing[:3]} in "
                         f"{root / spec.object_file}")
    return RawRelation(
        name=spec.relation_name,
        relation_type=spec.relation_type,
        templates=spec.templates,
        samples=tuple(
            (t[spec.subject_column], spec.object_phrase(phrase[t["region"]]))
            for t in towns
        ),
    )
