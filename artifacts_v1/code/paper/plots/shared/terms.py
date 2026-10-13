from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import json5_io
import relation_sets

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "metadata" / "probe-names.json5"
REF = re.compile(r"\{([A-Za-z0-9_$]+(?:\.[A-Za-z0-9_$]+)+)\}")
TERM_SETS = ("technical", "readable")
CASINGS = ("sentence", "title", "lower")
TEXT_NODE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S)

FIGURE_STYLES = ("B",)

class TermsError(ValueError):
    pass

def _fail(message: str) -> None:
    raise TermsError(f"terms: {message}")

class Terms:
    def __init__(self, path: Path, raw: bytes, sets: relation_sets.RelationSets):
        try:
            self.doc = json5_io.loads(raw.decode("utf-8"), str(path))
        except json5_io.Json5Error as error:
            _fail(str(error))
        if self.doc.get("schema") != "linroute/probe-names/1":
            _fail(f"{path} must declare schema linroute/probe-names/1")
        if not isinstance(self.doc.get("terms"), dict):
            _fail(f"{path} has no terms section")
        style = (self.doc.get("typography") or {}).get("probe_style")
        if style not in FIGURE_STYLES:
            _fail(f"{path} typography.probe_style is {style!r}; figures print capitals, so it must be one of {FIGURE_STYLES}")
        self.path = path
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.sets = sets
        t = self.doc["terms"]
        self.active = t.get("active")
        if self.active not in TERM_SETS:
            _fail(f"{path} terms.active must be one of {TERM_SETS}, not {self.active!r}")
        if t.get("readable_casing") not in CASINGS:
            _fail(f"{path} terms.readable_casing must be one of {CASINGS}")
        for name in (*TERM_SETS, "common"):
            if not isinstance(t.get(name), dict):
                _fail(f"{path} terms.{name} must be an object")
        if set(t["technical"]) != set(t["readable"]):
            _fail(f"terms.technical and terms.readable define different keys: {sorted(set(t['technical']) ^ set(t['readable']))}")
        for key in t["technical"]:
            if key in t["common"]:
                _fail(f"terms.{key} is in a named set and in common")
            tex = [{f: v for f, v in t[n][key].items() if f.startswith("tex_")} for n in TERM_SETS]
            if tex[0] != tex[1]:
                _fail(f"terms.{key}: the tex_ fields differ between the technical and readable sets")
        for name in (*TERM_SETS, "common"):
            for key, entry in t[name].items():
                if not isinstance(entry.get("plain"), str) or not entry["plain"]:
                    _fail(f"terms.{name}.{key} needs a non-empty plain form")
                for form, value in entry.items():
                    if isinstance(value, str) and not form.startswith("tex_"):
                        self.resolve(value)

    def casing(self, value: str) -> str:
        mode = self.doc["terms"]["readable_casing"]
        if mode == "lower":
            return value.lower()
        if mode == "title":
            return re.sub(r"(^|[\s-])([a-z])", lambda m: m.group(1) + m.group(2).upper(), value)
        return value

    def raw(self, path: str) -> str:
        keys = path.split(".")
        if keys[0] == "terms" and len(keys) == 4:
            t = self.doc["terms"]
            name = self.active if keys[1] == "$" else keys[1]
            if name not in (*TERM_SETS, "common"):
                _fail(f"{self.path} has no term set {keys[1]} (at {path})")
            entry = t[name].get(keys[2])
            if entry is None and keys[1] == "$":
                name, entry = "common", t["common"].get(keys[2])
            value = (entry or {}).get(keys[3])
            if not isinstance(value, str):
                _fail(f"{self.path} has no string at {path}")
            return self.casing(value) if name == "readable" and "{" not in value else value
        node = self.doc
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, str):
            _fail(f"{self.path} has no string at {path}")
        return node

    def resolve(self, value: str, stack: tuple[str, ...] = ()) -> str:
        def one(match: re.Match) -> str:
            path = match.group(1)
            if path in stack:
                _fail(f"circular reference {' -> '.join([*stack, path])}")
            return self.resolve(self.raw(path), (*stack, path))
        return REF.sub(one, value)

    def entry(self, key: str) -> dict:
        t = self.doc["terms"]
        entry = t[self.active].get(key) or t["common"].get(key)
        if entry is None:
            _fail(f"no term {key}")
        return entry

    def term(self, key: str, form: str = "plain") -> str:
        if not isinstance(self.entry(key).get(form), str):
            _fail(f"term {key} has no {form} form")
        return self.resolve(self.raw(f"terms.$.{key}.{form}"))

    def set_strings(self, name: str) -> set[str]:
        return {self.resolve(self.raw(f"terms.{name}.{key}.{form}"))
                for key, entry in self.doc["terms"][name].items()
                for form, value in entry.items() if isinstance(value, str) and not form.startswith("tex_")}

    def check_svg(self, svg: str, name: str = "svg") -> list[str]:
        inactive = TERM_SETS[1 - TERM_SETS.index(self.active)]
        banned = {s for s in self.set_strings(inactive) - self.set_strings(self.active)
                  if " " in s or "-" in s or not s.islower()}
        text = " | ".join(m.group(1) for m in TEXT_NODE.finditer(svg))
        return [f"{name}: prints the {inactive} term {s!r} while terms.active is {self.active}" for s in sorted(banned)
                if re.search(r"(?<![A-Za-z0-9-])" + re.escape(s) + r"(?![A-Za-z0-9])", text)]

    def at(self, path: str) -> str:
        return self.resolve(self.raw(path))

    def set_name(self, source: str) -> str:
        return self.sets.display_name(source)

def load(path: Path | str | None = None, sets: relation_sets.RelationSets | None = None) -> Terms:
    path = Path(path or os.environ.get("PROBE_NAMES_PATH") or DEFAULT_PATH)
    return Terms(path, path.read_bytes(), sets or relation_sets.load())
