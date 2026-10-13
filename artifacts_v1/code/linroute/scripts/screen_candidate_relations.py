#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import screen_relations as base

from linroute import relations

SOURCE = "external_pack"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def pack_candidates(relations_dir: Path) -> list[tuple[relations.RawRelation, str, dict]]:
    root = Path(relations_dir)
    out: list[tuple[relations.RawRelation, str, dict]] = []
    for path in sorted(root.glob("*/*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        props = blob.get("properties") or {}
        templates = tuple(blob.get("prompt_templates") or ())
        if not templates:
            raise SystemExit(f"{path}: no prompt_templates; cannot render a prompt")
        out.append((
            relations.RawRelation(
                name=blob["name"],
                relation_type=props.get("relation_type") or path.parent.name,
                templates=templates,
                samples=tuple(
                    (s["subject"], s["object"]) for s in blob.get("samples") or ()
                ),
            ),
            SOURCE,
            {
                "kind": "external_pack",
                "pack_root": str(root),
                "relation_file": str(path.relative_to(root)),
                "relation_sha256": _sha256(path),
                "declared_source": props.get("source") or {},
                "single_value_rule": props.get("single_value_rule"),
                "verified_against": None,
            },
        ))
    if not out:
        raise SystemExit(f"no <type>/<id>.json below {root}")
    return out

UNSCREENABLE: list[dict] = []

def preflight(pool, tokenizers, *, n_shots: int, require_context_consistent: bool):
    from linroute import admission, inventory
    from linroute.cache import slug

    ok, short = [], []
    for raw, source, provenance in pool:
        sizes = {}
        for hf_id, tok in tokenizers.items():
            inv = inventory.build_inventory(
                raw, tok, tokenizer_id=hf_id, relation_slug=slug(raw.name),
            )
            subjects, _, _ = admission.subjects_from_inventory(
                inv, require_context_consistent=require_context_consistent
            )
            sizes[hf_id] = len(subjects)
        if min(sizes.values()) < n_shots:
            short.append({
                "relation": raw.name,
                "relation_file": provenance.get("relation_file"),
                "relation_sha256": provenance.get("relation_sha256"),
                "n_samples": len(raw.samples),
                "eligible_pool_by_tokenizer": sizes,
                "n_shots": n_shots,
                "reason": "pool_smaller_than_support_reservation",
                "detail": (
                    "the eligible pool holds fewer subjects than the "
                    f"{n_shots} demonstrations reserve, so no manifest exists "
                    "to admit or refuse"
                ),
            })
        else:
            ok.append((raw, source, provenance))
    return ok, short

def main(argv: list[str] | None = None) -> int:
    from linroute import prompts

    args = base.parse_args(argv)
    tokenizers = base.load_tokenizers(args.tokenizers)
    n_shots = prompts.few_shot_spec().n_shots
    pool, short = preflight(
        pack_candidates(Path(args.relations_dir).expanduser()),
        tokenizers,
        n_shots=n_shots,
        require_context_consistent=args.require_context_consistent,
    )
    UNSCREENABLE.extend(short)

    base.candidates = lambda _dir: pool
    base.load_tokenizers = lambda _ids: tokenizers
    rc = base.main(argv)

    out = Path(args.out).expanduser().resolve() / "screen" / "unscreenable.json"
    out.write_text(
        json.dumps(
            {
                "schema": "linroute/admission/screen/unscreenable/1",
                "relations_dir": str(Path(args.relations_dir).expanduser()),
                "n_screened": len(pool),
                "n_unscreenable": len(UNSCREENABLE),
                "relations": UNSCREENABLE,
            },
            indent=2, ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"{len(UNSCREENABLE)} pack file(s) unscreenable; wrote {out}")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
