from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import json5_io
except SystemExit:
    json5_io = None
from typing import Iterable, Mapping, Sequence

SCHEMA = "linroute/relation-sets/1"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "metadata" / "relation-sets.json5"
_ID = re.compile(r"^[A-Za-z0-9_]+$")
_MARKS = {"triangle", "ring", "dot"}
_SELECTION_KEYS = {"note", "base", "sources", "flags", "include_retired", "include", "exclude"}

class RelationSetError(ValueError):
    pass

def _fail(message: str) -> None:
    raise RelationSetError(f"relation-sets: {message}")

def _strings(value, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence) or not all(isinstance(v, str) for v in value):
        _fail(f"{what} must be a list of strings")
    if len(set(value)) != len(value):
        _fail(f"{what} repeats an entry")
    return list(value)

def digest(ids: Iterable[str]) -> str:
    text = json.dumps(list(ids), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class RelationSets:
    def __init__(self, path: Path, raw: bytes):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        try:
            document = json5_io.loads(text, str(path)) if json5_io else json.loads(text)
        except (ValueError, json.JSONDecodeError) as error:
            _fail(f"cannot parse {path}: {error}")
        if not isinstance(document, dict) or document.get("schema") != SCHEMA:
            _fail(f"{path}: expected schema {SCHEMA}")
        self.path = path
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.sources = set(document.get("sources") or {})
        self.flags = set(document.get("flags") or {})
        self.relations: dict[str, dict] = document.get("relations") or {}
        self.selections: dict[str, dict] = document.get("selections") or {}
        self.aliases: dict[str, str] = {
            k: v for k, v in (document.get("aliases") or {}).items() if not k.startswith("_")}
        if not self.sources or not self.relations:
            _fail(f"{path}: empty sources or relations")

        self.source_display: dict[str, dict] = {
            k: v for k, v in (document.get("source_display") or {}).items() if not k.startswith("_")}
        for source in sorted(self.sources):
            name = (self.source_display.get(source) or {}).get("name")
            if not isinstance(name, str) or not name.strip():
                _fail(f"source_display.{source}.name is missing or empty")
            entry = self.source_display[source]
            if not re.fullmatch(r"#[0-9A-F]{6}", entry.get("color") or ""):
                _fail(f"source_display.{source}.color must be #RRGGBB (upper case)")
            if entry.get("mark") not in _MARKS:
                _fail(f"source_display.{source}.mark must be one of {', '.join(sorted(_MARKS))}")
        for key in self.source_display:
            if key not in self.sources:
                _fail(f"source_display names unknown source {key!r}")
        for rid, relation in self.relations.items():
            if not _ID.match(rid):
                _fail(f"relation id {rid!r} is not [A-Za-z0-9_]+")
            if relation.get("source") not in self.sources:
                _fail(f"relation {rid}: unknown source {relation.get('source')!r}")
            for flag in _strings(relation.get("flags"), f"relation {rid} flags"):
                if flag not in self.flags:
                    _fail(f"relation {rid}: unknown flag {flag!r}")
            if relation.get("retired", True) is not True:
                _fail(f"relation {rid}: retired must be true or absent")
        for alias, target in self.aliases.items():
            if not _ID.match(alias):
                _fail(f"alias {alias!r} is not [A-Za-z0-9_]+")
            if alias in self.relations:
                _fail(f"alias {alias} is itself a relation id")
            if target not in self.relations:
                _fail(f"alias {alias} points at unknown relation {target!r}")

        self._resolved: dict[str, list[str]] = {}
        for name in self.selections:
            self.named(name)

    def select(self, *, base: str | None = None, sources: Sequence[str] | None = None,
               flags: Sequence[str] | None = None, include: Sequence[str] | None = None,
               exclude: Sequence[str] | None = None, include_retired: bool = False,
               context: str = "select") -> list[str]:
        if base is not None:
            if sources is not None or flags is not None or include_retired:
                _fail(f"{context}: base excludes sources, flags and include_retired")
            members = set(self.named(base))
        else:
            wanted_sources = _strings(sources, f"{context} sources")
            for source in wanted_sources:
                if source not in self.sources:
                    _fail(f"{context}: unknown source {source!r}")
            wanted_flags = _strings(flags, f"{context} flags")
            for flag in wanted_flags:
                if flag not in self.flags:
                    _fail(f"{context}: unknown flag {flag!r}")
            source_set = self.sources if sources is None else set(wanted_sources)
            members = {
                rid for rid, relation in self.relations.items()
                if relation["source"] in source_set
                and all(flag in (relation.get("flags") or []) for flag in wanted_flags)
                and (include_retired or not relation.get("retired"))
            }
        for rid in _strings(include, f"{context} include"):
            if rid not in self.relations:
                _fail(f"{context}: unknown relation {rid!r} in include")
            if rid in members:
                _fail(f"{context}: include of {rid}, which is already a member")
            members.add(rid)
        for rid in _strings(exclude, f"{context} exclude"):
            if rid not in self.relations:
                _fail(f"{context}: unknown relation {rid!r} in exclude")
            if rid not in members:
                _fail(f"{context}: exclude of {rid}, which is not a member")
            members.discard(rid)
        ids = sorted(members)
        if not ids:
            _fail(f"{context}: empty selection")
        return ids

    def named(self, name: str, _stack: tuple[str, ...] = ()) -> list[str]:
        if name in getattr(self, "_resolved", {}):
            return list(self._resolved[name])
        definition = self.selections.get(name)
        if definition is None:
            _fail(f"unknown selection {name!r}")
        if name in _stack:
            _fail("selection cycle: " + " -> ".join((*_stack, name)))
        unknown = set(definition) - _SELECTION_KEYS
        if unknown:
            _fail(f"selection {name}: unknown key {sorted(unknown)[0]}")
        base = definition.get("base")
        if base is not None:
            self.named(base, (*_stack, name))
        ids = self.select(
            base=base, sources=definition.get("sources"), flags=definition.get("flags"),
            include=definition.get("include"), exclude=definition.get("exclude"),
            include_retired=definition.get("include_retired") is True,
            context=f"selection {name}",
        )
        self._resolved[name] = ids
        return list(ids)

    def canonical(self, rid: str) -> str:
        return self.aliases.get(rid, rid)

    def display_name(self, source: str) -> str:
        if source not in self.sources:
            _fail(f"unknown source {source!r}")
        return self.source_display[source]["name"]

    def display_color(self, source: str) -> str:
        if source not in self.sources:
            _fail(f"unknown source {source!r}")
        return self.source_display[source]["color"]

    def display_mark(self, source: str) -> str:
        if source not in self.sources:
            _fail(f"unknown source {source!r}")
        return self.source_display[source]["mark"]

    def source(self, rid: str) -> str:
        if rid not in self.relations:
            _fail(f"unknown relation {rid!r}")
        return self.relations[rid]["source"]

    def pinned(self, pin: Mapping, context: str) -> list[str]:
        if not isinstance(pin, Mapping) or not isinstance(pin.get("selection"), str) \
                or not isinstance(pin.get("digest"), str):
            _fail(f"{context}: a pin needs selection and digest")
        ids = self.named(pin["selection"])
        actual = digest(ids)
        if actual != pin["digest"]:
            _fail(f"{context}: selection {pin['selection']} digest is {actual}, the spec pins "
                  f"{pin['digest']}; its members changed, so re-render and re-pin deliberately")
        return ids

def load(path: Path | str = DEFAULT_PATH) -> RelationSets:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        _fail(f"cannot read {source}: {error}")
    return RelationSets(source, raw)

def resolve_spec(spec: dict, spec_path: Path | str) -> RelationSets:
    spec_path = Path(spec_path)
    block = spec.get("relation_registry")
    if not block:
        _fail(f"{spec_path}: no relation_registry block")
    if "relations" in spec or "relation_sets" in spec:
        _fail(f"{spec_path}: literal relations/relation_sets beside relation_registry")
    if not isinstance(block.get("path"), str):
        _fail(f"{spec_path}: relation_registry.path missing")
    if block.get("relations") is None and block.get("relation_sets") is None:
        _fail(f"{spec_path}: relation_registry pins neither relations nor relation_sets")
    registry = load(spec_path.parent / block["path"])
    palette = block.get("palette")
    if block.get("relations") is not None:
        ids = registry.pinned(block["relations"], f"{spec_path} relation_registry.relations")
        if palette is not None:
            if not isinstance(palette, list) or len(palette) != len(ids):
                _fail(f"{spec_path}: relation_registry.palette has "
                      f"{len(palette) if isinstance(palette, list) else palette!r} colors for {len(ids)} relations")
            spec["relations"] = [{"id": rid, "color": color} for rid, color in zip(ids, palette)]
        else:
            spec["relations"] = [{"id": rid} for rid in ids]
    elif palette is not None:
        _fail(f"{spec_path}: relation_registry.palette needs a relations pin")
    if "relation_sets" in block:
        spec["relation_sets"] = {
            key: registry.pinned(pin, f"{spec_path} relation_registry.relation_sets.{key}")
            for key, pin in block["relation_sets"].items()
        }
    return registry
