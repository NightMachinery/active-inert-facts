from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from linroute import relations as relmod
from linroute import spans

__all__ = [
    "SCHEMA",
    "GENERATOR_VERSION",
    "EXCLUSION_REASONS",
    "HISTORICAL_ROLES",
    "POLICIES",
    "Rendered",
    "Record",
    "EffectiveClass",
    "RelationInventory",
    "record_id",
    "subject_id",
    "build_inventory",
    "records_jsonl",
    "partition_signature",
    "compare_partitions",
]

SCHEMA = "linroute/relations/inventory/2"
GENERATOR_VERSION = "2.0.0"

EXCLUSION_REASONS = relmod.EXCLUSION_REASONS

POLICIES = {
    "token_rule": relmod.TOKEN_RULE,
    "scored_token": (
        'first NON-DEGENERATE token of " " + object, tokenized in isolation '
        "with add_special_tokens=False (linroute.relations.scored_token); the "
        "degenerate ids before it are the record's skipped_prefix_ids"
    ),
    "effective_class": (
        "objects sharing a scored token id are one class; merging is "
        "deliberate and is never undone by the rendered comparison"
    ),
    "rendered_context": (
        "the full prompt spans.fill(template, subject) + ' ' + object, "
        "tokenized with the tokenizer's DEFAULT add_special_tokens, answer "
        "span located by character-offset overlap (linroute.spans.encode_span)"
    ),
    "answer_boundary": (
        "the answer is the appended object, separated from the prompt by a "
        "single space; the rendered scored token is the one at "
        "span_start + scored_token_index, and the rendered ids before it "
        "inside the span must equal skipped_prefix_ids (prefix_matches)"
    ),
    "prefix": (
        "the readout is taken at the last PROMPT token with nothing appended, "
        "so candidates whose scored token sits after different skipped "
        "prefixes are not comparable; this inventory REPORTS that as "
        "prefix_uniform and never raises, while "
        "linroute.relations.prepare_relation refuses such a relation"
    ),
    "exact_duplicates": (
        "identical (subject, object) source rows are folded into one record "
        "that lists every source row index it came from"
    ),
    "duplicate_subjects": (
        "a subject string carrying more than one distinct object is excluded "
        "as unresolved_object_conflict, all of its rows, with no first-row-wins"
    ),
    "empty_objects": "a row whose object is blank after strip() is excluded",
    "degenerate_tokens": (
        "a token this tokenizer decodes to whitespace or to punctuation only "
        "is degenerate (linroute.relations._degenerate) and is skipped; digits "
        "count as content.  Only an object whose tokens are ALL degenerate is "
        "excluded, as all_tokens_degenerate"
    ),
    "entities": (
        "no entity resolution is performed; subject IDs hash subject strings, "
        "entity_id is null and class support counts raw subject strings"
    ),
    "exclusion_order": " -> ".join(EXCLUSION_REASONS),
    "sort_order": (
        "records sort by (scored token id, subject_id, record_id), with "
        "excluded records that have no token sorted last under token id null"
    ),
}

def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

def subject_id(relation: str, subject: str) -> str:
    return _digest("subject", relation, subject)[:16]

def record_id(relation: str, subject: str, obj: str) -> str:
    return _digest("record", relation, subject, obj)[:16]

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class Rendered:
    template_index: int
    template_role: str
    prompt_sha256: str
    answer_char_span: tuple[int, int]
    answer_token_span: tuple[int, int]
    answer_region_span: tuple[int, int]
    first_token_id: int
    first_token_str: str
    scored_token_id: int | None
    scored_token_str: str | None
    prefix_matches: bool
    n_prompt_tokens: int
    matches_isolated: bool
    subject_span_round_trips: bool
    answer_span_round_trips: bool

@dataclass(frozen=True)
class Record:
    record_id: str
    subject_id: str
    subject: str
    object: str
    source_rows: tuple[int, ...]
    status: str
    reason: str | None
    entity_id: None
    entity_status: str
    isolated_token_ids: tuple[int, ...]
    scored_token_id: int | None
    scored_token_str: str | None
    scored_token_index: int | None
    skipped_prefix_ids: tuple[int, ...]
    skipped_prefix_str: str
    n_object_tokens: int
    effective_class: int | None
    rendered: tuple[Rendered, ...]
    context_consistent: bool
    historical_role: str

    def sort_key(self) -> tuple:
        token = self.scored_token_id
        return (token is None, token if token is not None else 0,
                self.subject_id, self.record_id)

@dataclass(frozen=True)
class EffectiveClass:
    token_id: int
    token_str: str
    n_objects: int
    merged: bool
    support_subject_strings: int
    support_certified_entities: None
    n_context_consistent: int

@dataclass
class RelationInventory:
    relation: str
    relation_slug: str
    relation_type: str
    tokenizer: str
    records: list[Record] = field(default_factory=list)
    classes: list[EffectiveClass] = field(default_factory=list)
    templates: tuple[str, ...] = ()

    @property
    def n_source_rows(self) -> int:
        return sum(len(r.source_rows) for r in self.records)

    @property
    def n_records(self) -> int:
        return len(self.records)

    @property
    def included(self) -> list[Record]:
        return [r for r in self.records if r.status == "included"]

    @property
    def empty(self) -> bool:
        return not self.included

    def exclusion_counts(self) -> dict[str, int]:
        counts = Counter(r.reason for r in self.records if r.reason)
        return {reason: counts.get(reason, 0) for reason in EXCLUSION_REASONS}

    def skipped_prefixes(self) -> list[dict]:
        order: list[tuple[int, ...]] = []
        counts: Counter = Counter()
        decoded: dict[tuple[int, ...], str] = {}
        for r in self.included:
            if r.skipped_prefix_ids not in decoded:
                order.append(r.skipped_prefix_ids)
                decoded[r.skipped_prefix_ids] = r.skipped_prefix_str
            counts[r.skipped_prefix_ids] += 1
        return [
            {"ids": list(ids), "decoded": decoded[ids], "n_records": counts[ids]}
            for ids in order
        ]

    @property
    def prefix_uniform(self) -> bool:
        return len(self.skipped_prefixes()) <= 1

    def conflicting_subjects(self) -> list[str]:
        return sorted(
            {r.subject for r in self.records
             if r.reason == "unresolved_object_conflict"}
        )

    def summary(self, *, records_sha256: str | None = None) -> dict:
        supports = sorted(c.support_subject_strings for c in self.classes)
        included = self.included
        partition_hash, subjects_hash = partition_signature(self)
        return {
            "tokenizer": self.tokenizer,
            "empty": self.empty,
            "n_source_rows": self.n_source_rows,
            "n_records": self.n_records,
            "n_exact_duplicate_rows": self.n_source_rows - self.n_records,
            "n_included": len(included),
            "n_excluded": self.n_records - len(included),
            "excluded_by_reason": self.exclusion_counts(),
            "n_conflicting_subject_strings": len(self.conflicting_subjects()),
            "skipped_prefixes": self.skipped_prefixes(),
            "prefix_uniform": self.prefix_uniform,
            "n_context_mismatched": sum(
                1 for r in included if not r.context_consistent
            ),
            "n_context_consistent": sum(
                1 for r in included if r.context_consistent
            ),
            "n_subject_span_round_trip_failures": sum(
                1 for r in self.records for e in r.rendered
                if not e.subject_span_round_trips
            ),
            "n_answer_span_round_trip_failures": sum(
                1 for r in self.records for e in r.rendered
                if not e.answer_span_round_trips
            ),
            "n_objects": len({r.object for r in included}),
            "n_effective_classes": len(self.classes),
            "n_merged_classes": sum(1 for c in self.classes if c.merged),
            "n_single_object_classes": sum(
                1 for c in self.classes if not c.merged
            ),
            "n_multi_token_objects": len(
                {r.object for r in included if r.n_object_tokens > 1}
            ),
            "support_subject_strings": {
                "min": supports[0] if supports else None,

                "median": supports[(len(supports) - 1) // 2] if supports else None,
                "max": supports[-1] if supports else None,
            },
            "support_certified_entities": None,
            "historical_roles": dict(
                sorted(Counter(r.historical_role for r in self.records).items())
            ),
            "partition_hash": partition_hash,
            "subjects_hash": subjects_hash,
            "classes": [asdict(c) for c in self.classes],
            "records_sha256": records_sha256,
        }

def _fold_duplicates(
    samples: Sequence[tuple[str, str]]
) -> list[tuple[str, str, tuple[int, ...]]]:
    order: list[tuple[str, str]] = []
    rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, pair in enumerate(samples):
        if pair not in rows:
            order.append(pair)
        rows[pair].append(i)
    return [(s, o, tuple(rows[(s, o)])) for s, o in order]

HISTORICAL_ROLES = (
    "evaluation",
    "few_shot_support",
    "unreached",
    "filtered_out",
)

def _historical_roles(
    raw: relmod.RawRelation, tokenizer, *, subject_cap: int, seed: int
) -> dict[str, str]:
    capped = relmod.prepare_relation(
        raw, tokenizer, subject_cap=subject_cap, seed=seed,
        require_uniform_prefix=False,
    )
    uncapped = relmod.prepare_relation(
        raw, tokenizer, subject_cap=len(raw.samples) + 1, seed=seed,
        require_uniform_prefix=False,
    )
    roles = {s: "filtered_out" for s, _ in raw.samples}
    roles.update({s: "unreached" for s in uncapped.subjects})
    roles.update({s: "evaluation" for s in capped.subjects})
    roles.update({s: "few_shot_support" for s, _ in capped.support})
    return roles

def build_inventory(
    raw: relmod.RawRelation,
    tokenizer,
    *,
    tokenizer_id: str,
    relation_slug: str,
    historical_cap: int = 300,
    historical_seed: int = 0,
    templates: Sequence[int] | None = None,
) -> RelationInventory:
    folded = _fold_duplicates(raw.samples)
    indices = range(len(raw.templates)) if templates is None else templates

    objects_by_subject: dict[str, set[str]] = defaultdict(set)
    for subject, obj, _ in folded:
        if obj.strip():
            objects_by_subject[subject].add(obj)
    conflicted = {s for s, objs in objects_by_subject.items() if len(objs) > 1}

    roles = _historical_roles(
        raw, tokenizer, subject_cap=historical_cap, seed=historical_seed
    )

    inv = RelationInventory(
        relation=raw.name,
        relation_slug=relation_slug,
        relation_type=raw.relation_type,
        tokenizer=tokenizer_id,
        templates=tuple(raw.templates),
    )

    for subject, obj, rows in folded:
        ids: tuple[int, ...] = ()
        scored: int | None = None
        scored_str: str | None = None
        scored_index: int | None = None
        prefix: tuple[int, ...] = ()
        reason: str | None = None

        if not obj.strip():
            reason = "empty_object"
        else:
            selected = relmod.scored_token(tokenizer, obj)
            ids = selected.ids
            scored = selected.token_id
            scored_index = selected.index
            prefix = selected.skipped_prefix
            scored_str = None if scored is None else tokenizer.decode([scored])
            if subject in conflicted:
                reason = "unresolved_object_conflict"
            elif selected.all_degenerate:
                reason = "all_tokens_degenerate"

        rendered = (
            ()
            if not obj.strip()
            else tuple(
                _render(tokenizer, raw.templates[i], i, subject, obj, scored,
                        prefix)
                for i in indices
            )
        )
        canonical = [e for e in rendered if e.template_role == "canonical"]

        consistent = bool(canonical) and all(
            e.matches_isolated and e.prefix_matches for e in canonical
        )

        inv.records.append(
            Record(
                record_id=record_id(raw.name, subject, obj),
                subject_id=subject_id(raw.name, subject),
                subject=subject,
                object=obj,
                source_rows=rows,
                status="excluded" if reason else "included",
                reason=reason,
                entity_id=None,
                entity_status="unresolved",
                isolated_token_ids=ids,
                scored_token_id=scored,
                scored_token_str=scored_str,
                scored_token_index=scored_index,
                skipped_prefix_ids=prefix,
                skipped_prefix_str=tokenizer.decode(list(prefix)) if prefix else "",
                n_object_tokens=len(ids),
                effective_class=None if reason else scored,
                rendered=rendered,
                context_consistent=consistent and not reason,
                historical_role=roles[subject],
            )
        )

    inv.records.sort(key=Record.sort_key)
    inv.classes = _classes(inv, tokenizer)
    return inv

def _render(
    tokenizer, template: str, index: int, subject: str, obj: str,
    scored: int | None, prefix: tuple[int, ...] = (),
) -> Rendered:
    text, s_start, s_end = spans.fill(template, subject)
    subject_enc = spans.encode_span(tokenizer, text, s_start, s_end)
    forced, a_start, a_end = spans.append(text, obj)
    answer_enc = spans.encode_span(tokenizer, forced, a_start, a_end)
    region = spans.encode_span(tokenizer, forced, a_start - 1, a_end)
    start, end = region.span
    first = answer_enc.ids[answer_enc.span[0]]

    at = start + len(prefix)
    rendered_scored = int(region.ids[at]) if at < end else None
    return Rendered(
        template_index=index,
        template_role="canonical" if index == 0 else "secondary",
        prompt_sha256=_sha256_text(forced),
        answer_char_span=(a_start, a_end),
        answer_token_span=answer_enc.span,
        answer_region_span=region.span,
        first_token_id=int(first),
        first_token_str=tokenizer.decode([first]),
        scored_token_id=rendered_scored,
        scored_token_str=(
            None if rendered_scored is None else tokenizer.decode([rendered_scored])
        ),
        prefix_matches=tuple(
            int(i) for i in region.ids[start : min(at, end)]
        ) == tuple(prefix),
        n_prompt_tokens=len(answer_enc.ids),
        matches_isolated=scored is not None and rendered_scored == scored,
        subject_span_round_trips=spans.span_matches(
            tokenizer, subject_enc, subject
        ),
        answer_span_round_trips=spans.span_matches(tokenizer, answer_enc, obj),
    )

def _classes(inv: RelationInventory, tokenizer) -> list[EffectiveClass]:
    objects: dict[int, set[str]] = defaultdict(set)
    subjects: dict[int, set[str]] = defaultdict(set)
    consistent: Counter = Counter()
    for r in inv.included:
        objects[r.scored_token_id].add(r.object)
        subjects[r.scored_token_id].add(r.subject)
        consistent[r.scored_token_id] += int(r.context_consistent)
    return [
        EffectiveClass(
            token_id=token,
            token_str=tokenizer.decode([token]),
            n_objects=len(objects[token]),
            merged=len(objects[token]) > 1,
            support_subject_strings=len(subjects[token]),
            support_certified_entities=None,
            n_context_consistent=consistent[token],
        )
        for token in sorted(objects)
    ]

def partition_signature(inv: RelationInventory) -> tuple[str, str]:
    groups: dict[int, set[str]] = defaultdict(set)
    for r in inv.included:
        groups[r.scored_token_id].add(r.subject_id)
    canonical = sorted(tuple(sorted(g)) for g in groups.values())
    subjects = sorted({r.subject_id for r in inv.included})
    return (
        hashlib.sha256(repr(canonical).encode()).hexdigest()[:16],
        hashlib.sha256(repr(subjects).encode()).hexdigest()[:16],
    )

def compare_partitions(inventories: Iterable[RelationInventory]) -> dict:
    invs = list(inventories)
    common: set[str] | None = None
    for inv in invs:
        ids = {r.subject_id for r in inv.included}
        common = ids if common is None else (common & ids)
    common = common or set()

    def restricted(inv: RelationInventory) -> str:
        groups: dict[int, set[str]] = defaultdict(set)
        for r in inv.included:
            if r.subject_id in common:
                groups[r.scored_token_id].add(r.subject_id)
        canonical = sorted(tuple(sorted(g)) for g in groups.values())
        return hashlib.sha256(repr(canonical).encode()).hexdigest()[:16]

    hashes = {inv.tokenizer: restricted(inv) for inv in invs}
    agreeing: dict[str, list[str]] = defaultdict(list)
    for name, h in hashes.items():
        agreeing[h].append(name)
    return {
        "n_common_subjects": len(common),

        "vacuous": not common,
        "identical_subject_sets": len(
            {partition_signature(inv)[1] for inv in invs}
        ) == 1,
        "restricted_partition_hash": dict(sorted(hashes.items())),
        "agree_up_to_class_names": len(set(hashes.values())) <= 1,
        "groups": [sorted(v) for _, v in sorted(agreeing.items())],
        "n_classes": {
            inv.tokenizer: len(inv.classes) for inv in sorted(
                invs, key=lambda i: i.tokenizer
            )
        },
    }

def records_jsonl(inv: RelationInventory) -> str:
    lines = []
    for r in inv.records:
        blob = asdict(r)
        blob["source_rows"] = list(r.source_rows)
        blob["isolated_token_ids"] = list(r.isolated_token_ids)
        blob["skipped_prefix_ids"] = list(r.skipped_prefix_ids)
        blob["rendered"] = [
            {**asdict(e),
             "answer_char_span": list(e.answer_char_span),
             "answer_token_span": list(e.answer_token_span),
             "answer_region_span": list(e.answer_region_span)}
            for e in r.rendered
        ]
        lines.append(json.dumps(blob, ensure_ascii=False, separators=(",", ":")))
    return "".join(line + "\n" for line in lines)
