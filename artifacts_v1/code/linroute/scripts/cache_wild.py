#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import admission
from linroute import cache as cachelib
from linroute import engineered_relations as engineered_rel
from linroute import models, prompts, relations, spans
from linroute.provenance import git_commit

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

RELATION_DIR_ENV = "LINROUTE_RELATIONS_DIR"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF id, e.g. allenai/Olmo-3-1025-7B")
    p.add_argument("--out", required=True, help="cache root directory")
    p.add_argument(
        "--relations",
        default="all",
        help="'all' or a comma-separated list of relation names",
    )
    p.add_argument(
        "--subject-cap", type=int, default=300,
        help="max subjects kept.  IGNORED when --split-manifest is given: the "
        "manifest already names every subject and its role",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for the subject shuffle (cap and few-shot support set)")
    p.add_argument("--batch-size", type=int, default=16, help="prompts per forward")
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, mps")
    p.add_argument(
        "--relations-dir",
        default=None,
        help=f"the Hernandez checkout's data directory, holding */*.json.  "
        f"Required; falls back to ${RELATION_DIR_ENV}.  Its prompts must "
        f"already be snapshotted in data/prompts/wild.yaml "
        f"(scripts/snapshot_wild_prompts.py)",
    )
    p.add_argument(
        "--no-require-fidelity",
        dest="require_fidelity",
        action="store_false",
        help="do not abort when the decode-fidelity check fails.  Off by "
        "default: a FAIL means the cached answer state does not reproduce the "
        "model's own logits, so every probe downstream reads the wrong vector",
    )
    p.add_argument(
        "--no-require-identity",
        dest="require_identity",
        action="store_false",
        help="extend a cache directory whose --subject-cap, --seed, --dtype, "
        "--batch-size, relation source or schema differs from the files "
        "already in it.  Off by default: mixing two protocols in one "
        "directory makes the table that reads it meaningless",
    )
    p.add_argument(
        "--split-manifest",
        default=None,
        help="a linroute/admission/2 split manifest "
        "(scripts/make_split_manifest.py).  Replaces the seeded cap and "
        "support draw: the evaluation subjects become the manifest's "
        "fine-tuning set followed by its held-out set, in manifest order, and "
        "the demonstrations become its reserved support.  Exactly one "
        "relation may be selected, and it must be the manifest's",
    )
    p.add_argument(
        "--allow-boundary-mismatch",
        action="store_true",
        help="write a relation whose rendered answer boundary disagrees with "
        "the isolated scored token for some subject, recording the counts in "
        "meta.  Off by default: the probe scores the isolated id, so a "
        "disagreement means the id scored is not the id the model emits there",
    )
    p.add_argument(
        "--no-town-region",
        dest="town_region",
        action="store_false",
        help="skip the sandbox town -> region relation.  It is on by default "
        "and is built inside this repository from data/prompts/sandbox.yaml "
        "and the two TSVs in data/sandbox/, so it needs no external checkout",
    )
    p.add_argument(
        "--engineered",
        dest="engineered",
        action="store_true",
        default=True,
        help="include the engineered town relations "
        f"({', '.join(engineered_rel.names())}).  On by default, like "
        "--town-region: they are built inside this repository from the frozen "
        "records in data/relations/engineered/ and need no external checkout",
    )
    p.add_argument(
        "--no-engineered",
        dest="engineered",
        action="store_false",
        help="skip the engineered town relations",
    )
    p.add_argument(
        "--weights-dir",
        default=None,
        help="cache a FINE-TUNED model instead of the pretrained one: a local "
        "HF save_pretrained directory (config.json + *.safetensors) with a "
        "MANIFEST.json listing every file's name, bytes and sha256 (written by "
        "scripts/train_arm.py --save-final-weights).  --model stays the BASE "
        "model's HF id: it supplies the tokenizer, the split-manifest "
        "tokenizer check and the cache directory name, so the file lands at "
        "the same <out>/<model_slug>/<relation>.pt path, with the same layers "
        "and positions, as the pretrained wild cache.  Every file is checked "
        "against MANIFEST.json before loading; the MANIFEST's sha256 goes into "
        "meta['weights_source'], the directory's _manifest.json and "
        "config_hash.  Off by default; when off, nothing about the run, its "
        "hash or its files changes",
    )
    args = p.parse_args(argv)

    args.split_manifest_provenance = None
    args.engineered_provenance = None
    args.weights_provenance = None
    if not args.relations_dir:
        args.relations_dir = os.environ.get(RELATION_DIR_ENV)
    needed = from_the_checkout(args.relations)
    if needed is not None and not args.relations_dir:
        raise missing_relations_dir(needed)
    return args

def repository_relation_names() -> tuple[str, ...]:
    return (prompts.sandbox_prompts().town_region.relation_name,
            *engineered_rel.names())

def from_the_checkout(selection: str) -> str | None:
    if selection == "all":
        return "--relations all"
    local = set(repository_relation_names())
    outside = [n.strip() for n in selection.split(",") if n.strip() not in local]
    return ("relations " + ", ".join(repr(n) for n in outside)
            if outside else None)

def missing_relations_dir(wanted: str) -> SystemExit:
    return SystemExit(
        f"--relations-dir is required (or set ${RELATION_DIR_ENV}) for "
        f"{wanted}: the Hernandez et al. (2024) relation JSONs are an external "
        f"checkout, not part of this repository, and there is deliberately no "
        f"default path.  Point it at the directory whose */*.json files are "
        f"snapshotted in data/prompts/wild.yaml.  A run of only the "
        f"repository-built relations "
        f"({', '.join(repr(n) for n in repository_relation_names())}) needs no "
        f"checkout and no flag."
    )

def collect_relations(args: argparse.Namespace) -> list[relations.RawRelation]:
    built: list[relations.RawRelation] = []
    provenance: dict[str, dict] = {}
    if args.town_region:
        built.append(relations.town_region_relation())
    if args.engineered:
        for entry in engineered_rel.build_all():
            built.append(entry.relation)
            provenance[entry.name] = dict(entry.provenance)

    wanted = (None if args.relations == "all"
              else [name.strip() for name in args.relations.split(",")])
    needed = from_the_checkout(args.relations)

    raws: list[relations.RawRelation] = []
    if needed is not None:
        if not args.relations_dir:
            raise missing_relations_dir(needed)
        root = Path(args.relations_dir).expanduser()
        if not root.is_dir():
            raise SystemExit(f"--relations-dir {root} is not a directory")
        raws = relations.load_relations(root)
        from_checkout = {r.name for r in raws}
        clash = sorted({r.name for r in built} & from_checkout)
        if clash:
            raise SystemExit(
                f"{root} already contains {len(clash)} relation(s) this "
                f"repository builds itself: {', '.join(repr(c) for c in clash)}."
                f"  The sandbox relation comes from data/prompts/sandbox.yaml "
                f"and data/sandbox/*.tsv and the engineered ones from the "
                f"frozen records in data/relations/engineered/; remove the "
                f"copies from the checkout, or pass --no-town-region / "
                f"--no-engineered."
            )
    raws = raws + built

    if wanted is None:
        selected = raws
    else:
        by_name = {r.name: r for r in raws}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise SystemExit(f"unknown relations: {missing}")
        selected = [by_name[name] for name in wanted]

    chosen = {r.name for r in selected}
    args.engineered_provenance = {
        name: block for name, block in provenance.items() if name in chosen
    } or None
    return selected

def load_split_manifest(
    args: argparse.Namespace, raws: list[relations.RawRelation]
) -> tuple[dict | None, dict | None]:
    if not args.split_manifest:
        return None, None
    path = Path(args.split_manifest).expanduser()
    if not path.is_file():
        raise SystemExit(f"--split-manifest {path} is not a file")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != relations.ADMISSION_SCHEMA:
        raise SystemExit(
            f"{path}: schema is {manifest.get('schema')!r}, not "
            f"{relations.ADMISSION_SCHEMA!r}.  Regenerate it with "
            f"scripts/make_split_manifest.py."
        )
    wanted = manifest.get("relation")
    if len(raws) != 1:
        raise SystemExit(
            f"--split-manifest selects the subjects of ONE relation "
            f"({wanted!r}), but {len(raws)} relations are selected.  Pass "
            f"--relations {wanted!r} (and --no-town-region if it is not the "
            f"sandbox relation), or drop --split-manifest: a directory holding "
            f"both manifest-selected and seeded-draw files records nowhere "
            f"which row came from which."
        )
    if raws[0].name != wanted:
        raise SystemExit(
            f"--split-manifest describes relation {wanted!r}, but this run "
            f"caches {raws[0].name!r}."
        )
    if manifest.get("tokenizer") != args.model:
        report = (
            f"{path}: built for tokenizer {manifest.get('tokenizer')!r}, this "
            f"run is {args.model!r}.  An effective class is a scored token id, "
            f"so a manifest from another tokenizer names classes this run does "
            f"not have."
        )
        if args.require_identity:
            raise SystemExit(report)
        print(f"WARNING: {report}", flush=True)
    provenance = {
        "path": args.split_manifest,
        "resolved_path": str(path),
        "sha256": cachelib.sha256_file(path),
        "schema": manifest.get("schema"),
        "relation": wanted,
        "tokenizer": manifest.get("tokenizer"),
        "n_support": len(manifest["support"]["subjects"]),
        "n_train": len(manifest["train"]["subjects"]),
        "n_holdout": len(manifest["holdout"]["subjects"]),

        "holdout_rule": admission.holdout_rule_of(manifest),
        "holdout_per_class": {str(c): n for c, n in
                              admission.per_class_holdout(manifest).items()},

        "subject_cap_ignored": True,
    }
    return manifest, provenance

def prepare(
    raw: relations.RawRelation, tokenizer, args: argparse.Namespace,
    manifest: dict | None,
) -> relations.Relation:
    try:
        return relations.prepare_relation(
            raw, tokenizer, subject_cap=args.subject_cap, seed=args.seed,
            manifest=manifest,
        )
    except relations.NonUniformPrefixError as exc:
        raise SystemExit(
            f"{exc}\nNothing further was written.  Drop the relation from "
            f"--relations, or change the answer strings so that one prefix "
            f"covers all of them."
        ) from exc
    except relations.ManifestMismatchError as exc:
        raise SystemExit(
            f"{exc}\nNothing was written.  The manifest was frozen against a "
            f"particular inventory; regenerate it with "
            f"scripts/make_split_manifest.py against the inventory this "
            f"tokenizer produces, rather than caching a split that names "
            f"subjects or classes this run does not have."
        ) from exc

def build_encodings(
    tokenizer, rel: relations.Relation
) -> tuple[list[spans.Encoded], list[spans.Encoded], list[spans.Encoded],
           list[spans.Encoded], dict]:
    rel_enc, neu_enc, gold_enc, gold_region = [], [], [], []
    subject_bad = object_bad = 0
    for subject, obj in zip(rel.subjects, rel.objects):
        for index, template in enumerate(rel.templates):
            text, start, end = spans.fill(template, subject)
            enc = spans.encode_span(tokenizer, text, start, end)
            subject_bad += not spans.span_matches(tokenizer, enc, subject)
            rel_enc.append(enc)
            forced, gstart, gend = spans.append(text, obj)
            gold = spans.encode_span(tokenizer, forced, gstart, gend)
            object_bad += not spans.span_matches(tokenizer, gold, obj)
            gold_enc.append(gold)
            if index == 0:
                gold_region.append(
                    spans.encode_span(tokenizer, forced, gstart - 1, gend)
                )
        for frame in prompts.neutral_templates():
            text, start, end = spans.fill(frame, subject)
            enc = spans.encode_span(tokenizer, text, start, end)
            subject_bad += not spans.span_matches(tokenizer, enc, subject)
            neu_enc.append(enc)
    checks = {
        "subject_span_mismatches": int(subject_bad),
        "object_span_mismatches": int(object_bad),
    }
    return rel_enc, neu_enc, gold_enc, gold_region, checks

def scored_boundary(
    tokenizer, rel: relations.Relation, gold_region: list[spans.Encoded],
    args: argparse.Namespace,
) -> torch.Tensor:
    rendered: list[int] = []
    offenders: list[str] = []
    for i, subject in enumerate(rel.subjects):
        enc = gold_region[i]
        start, end = enc.span
        at = start + rel.object_scored_index[i]
        token = int(enc.ids[at]) if at < end else -1
        prefix = tuple(int(x) for x in enc.ids[start : min(at, end)])
        rendered.append(token)
        if token != rel.object_first_token[i] or prefix != rel.common_skipped_prefix:
            offenders.append(
                f"{subject!r} -> {rel.objects[i]!r}: rendered "
                f"{tokenizer.decode([token]) if token >= 0 else '<span too short>'!r}"
                f" after prefix {[tokenizer.decode([x]) for x in prefix]}, "
                f"isolated {tokenizer.decode([rel.object_first_token[i]])!r} "
                f"after {[tokenizer.decode([x]) for x in rel.common_skipped_prefix]}"
            )
    if offenders and not args.allow_boundary_mismatch:
        raise SystemExit(
            f"rendered answer boundary disagrees with the isolated scored "
            f"token on relation {rel.name!r} of {args.model}, for "
            f"{len(offenders)} of {len(rel.subjects)} subjects:\n  "
            + "\n  ".join(offenders[:10])
            + (f"\n  ... and {len(offenders) - 10} more" if len(offenders) > 10
               else "")
            + "\nThe probes score the isolated id, so these subjects would be "
            "scored on a token the model does not emit there.  Nothing was "
            "written.  Re-run with --allow-boundary-mismatch only to "
            "investigate; the counts are then recorded in meta."
        )
    return torch.tensor(rendered, dtype=torch.long)

def scored_generation(tokenizer, generated: list[tuple[int, ...]]) -> torch.Tensor:
    out = []
    for ids in generated:
        selected = next(
            (i for i in ids if not relations._degenerate(tokenizer, i)), None
        )
        out.append(int(selected if selected is not None else ids[0]))
    return torch.tensor(out, dtype=torch.long)

def check_fidelity(
    model,
    tokenizer,
    rel: relations.Relation,
    rel_enc: list[spans.Encoded],
    args: argparse.Namespace,
) -> dict:
    stats = models.decode_fidelity(
        model,
        tokenizer,
        rel_enc[: min(8, args.batch_size)],
        torch.tensor(rel.cand_first_token, dtype=torch.long),
        device=args.device,
    )
    print(f"  {models.fidelity_summary(stats)}", flush=True)
    if not stats["passed"] and args.require_fidelity:
        raise SystemExit(
            f"decode fidelity FAILED on relation {rel.name!r} of {args.model}: "
            f"{stats}.  The cached answer state does not reproduce the model's "
            f"own logits, so the layer convention or the norm parameterization "
            f"is wrong for this architecture and every cached state would be "
            f"unusable.  Nothing further was written.  Re-run with "
            f"--no-require-fidelity only to investigate."
        )
    return stats

def check_serialized(
    live: dict,
    states: models.SpanStates,
    u_cand: torch.Tensor,
    norm_gain: torch.Tensor,
    eps: float,
    rel: relations.Relation,
    args: argparse.Namespace,
) -> dict:
    stats = {
        **live,
        **models.serialized_decode_stats(
            states.final, states.cand_logits, u_cand, norm_gain, eps
        ),
    }
    stats["passed"] = models.fidelity_passed(stats)
    print(f"  serialized: {models.fidelity_summary(stats)}", flush=True)
    if not stats["passed"] and args.require_fidelity:
        raise SystemExit(
            f"decode fidelity FAILED on the SERIALIZED states of relation "
            f"{rel.name!r} of {args.model}: {stats}.  The fp16 answer states "
            f"and head about to be written do not decode to what the model "
            f"itself chose, and by more than a tie, so every probe reading "
            f"this file would read a different decoder from the model's.  "
            f"Nothing was written.  Re-run with --no-require-fidelity only to "
            f"investigate."
        )
    return stats

def cache_relation(
    model,
    tokenizer,
    rel: relations.Relation,
    *,
    n_layers: int,
    d_model: int,
    args: argparse.Namespace,
) -> dict:
    n, t = len(rel.subjects), len(rel.templates)
    cand_ids = torch.tensor(rel.cand_first_token, dtype=torch.long)
    rel_enc, neu_enc, gold_enc, gold_region, checks = build_encodings(tokenizer, rel)

    fidelity = check_fidelity(model, tokenizer, rel, rel_enc, args)

    def run(encoded: list[spans.Encoded], name: str) -> models.SpanStates:
        return models.span_states(
            model,
            tokenizer,
            encoded,
            cand_ids,
            n_layers=n_layers,
            d_model=d_model,
            batch_size=args.batch_size,
            device=args.device,
            name=name,
        )

    rel_states = run(rel_enc, "rel_subject")
    neu_states = run(neu_enc, "neutral_subject")
    gold_states = run(gold_enc, "rel_gold_object")

    fs_prompts = [
        relations.few_shot_prompt(rel.templates[0], rel.support, subject)
        for subject in rel.subjects
    ]
    _, fs_text, fs_generated = models.greedy_generate(
        model, tokenizer, fs_prompts, batch_size=args.batch_size, device=args.device
    )
    fs_token = scored_generation(tokenizer, fs_generated)

    rendered_first = scored_boundary(tokenizer, rel, gold_region, args)
    isolated_first = torch.tensor(rel.object_first_token, dtype=torch.long)
    cand_n_tokens = torch.tensor(rel.cand_n_tokens, dtype=torch.long)
    object_n_tokens = torch.tensor(rel.object_n_tokens, dtype=torch.long)
    profile = models.tokenizer_profile(tokenizer)
    u_cand, norm_gain, norm_eps = models.decode_head(model, cand_ids)

    head_u16 = u_cand.to(torch.float16)
    head_gain16 = norm_gain.to(torch.float16)
    fidelity = check_serialized(
        fidelity, rel_states, head_u16, head_gain16, norm_eps, rel, args
    )
    layers = n_layers + 1

    return {
        "meta": {
            "schema_version": cachelib.SCHEMA_VERSION,
            "model": args.model,
            "model_revision": models.model_revision(model),
            "tokenizer": tokenizer.name_or_path,
            "relation": rel.name,
            "relation_type": rel.relation_type,
            "n_layers": n_layers,
            "d_model": d_model,
            "subject_cap": args.subject_cap,
            "seed": args.seed,
            "dtype_run": args.dtype,
            "created": datetime.datetime.now(datetime.UTC).isoformat(),
            "git_commit": git_commit(),
            "prompt_mode": "raw",
            "norm_kind": models.norm_kind(model),

            "config_hash": config_hash(args),

            "prompt_files": prompts.digests(),

            "split_manifest": args.split_manifest_provenance,

            "engineered_source": (args.engineered_provenance or {}).get(rel.name),
            "decode_fidelity": fidelity,
            "fewshot_support": [s for s, _ in rel.support],
            "cand_collision_rate": relations.collision_rate(rel.cand_first_token),
            "first_token_mismatch_count": int((rendered_first != isolated_first).sum()),
            "first_token_mismatch_rate": float(
                (rendered_first != isolated_first).float().mean()
            ),
            "multi_token_object_rate": float(
                (object_n_tokens > 1).float().mean()
            ),

            "token_rule": cachelib.TOKEN_RULE,
            "n_merged_classes": rel.n_merged_classes,
            "merged_class_rate": rel.merged_class_rate,
            "exclusions": [
                {"subject": e.subject, "object": e.object, "reason": e.reason}
                for e in rel.exclusions
            ],
            "n_excluded_by_reason": rel.excluded_by_reason(),
            "common_skipped_prefix": list(rel.common_skipped_prefix),
            "tokenizer_profile": profile,
            "tokenizer_summary": models.tokenizer_summary(profile),
            "adds_bos": profile["adds_bos"],
            "bos_id": profile["bos_id"],
            "neutral_first_token_has_context": profile["adds_bos"],
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            **checks,
        },
        "subjects": list(rel.subjects),
        "objects": list(rel.objects),
        "object_first_token": rendered_first,
        "object_first_token_isolated": isolated_first,
        "candidates": list(rel.candidates),
        "cand_objects": [list(group) for group in rel.cand_objects],
        "cand_first_token": cand_ids,
        "cand_n_tokens": cand_n_tokens,
        "cand_scored_index": torch.tensor(rel.cand_scored_index, dtype=torch.long),
        "cand_index": torch.tensor(rel.cand_index, dtype=torch.long),
        "object_scored_index": torch.tensor(rel.object_scored_index, dtype=torch.long),
        "object_n_tokens": object_n_tokens,
        "rel_templates": list(rel.templates),
        "neutral_templates": list(prompts.neutral_templates()),
        "rel_subject_last": rel_states.last.view(n, t, layers, d_model),
        "rel_subject_mean": rel_states.mean.view(n, t, layers, d_model),
        "neutral_subject_last": neu_states.last.view(n, 2, layers, d_model),
        "neutral_subject_mean": neu_states.mean.view(n, 2, layers, d_model),
        "rel_answer_final": rel_states.final.view(n, t, d_model),
        "rel_gold_object_mean": gold_states.mean.view(n, t, layers, d_model),
        "zs_top1": rel_states.top1.view(n, t),
        "zs_cand": rel_states.cand_top1.view(n, t),
        "fs_pred_token": fs_token,
        "fs_generation": fs_text,

        "fs_correct": fs_token == isolated_first,
        "head": {
            "U_cand": head_u16,
            "norm_gain": head_gain16,
            "norm_eps": norm_eps,
            "U_full_path": None,
        },
        "_complete": True,
    }

_OPTIONAL_ARGS = ("weights_dir", "weights_provenance")

def run_args(args: argparse.Namespace) -> dict:
    return {key: value for key, value in vars(args).items()
            if not (key in _OPTIONAL_ARGS and value is None)}

def load_finetuned(args: argparse.Namespace):
    torch.backends.cuda.enable_cudnn_sdp(False)
    base = transformers.AutoConfig.from_pretrained(args.model)
    config = transformers.AutoConfig.from_pretrained(args.weights_dir)
    base_text, text = base.get_text_config(), config.get_text_config()
    drift = {
        key: (getattr(base_text, key, None), getattr(text, key, None))
        for key in ("num_hidden_layers", "hidden_size", "vocab_size")
        if getattr(base_text, key, None) != getattr(text, key, None)
    }
    if list(getattr(base, "architectures", None) or []) != \
            list(getattr(config, "architectures", None) or []):
        drift["architectures"] = (base.architectures, config.architectures)
    if drift:
        raise SystemExit(
            f"--weights-dir {args.weights_dir} is not a fine-tune of "
            f"{args.model}: (base, fine-tuned) differ on {drift}.  Nothing was "
            f"written.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    models.check_tokenizer(tokenizer, args.model)
    model = models._model_class(config).from_pretrained(
        args.weights_dir, dtype=DTYPES[args.dtype])
    model.to(args.device).eval()
    return model, tokenizer, text.num_hidden_layers, text.hidden_size

def manifest_path(root: Path, args: argparse.Namespace) -> Path:
    return root / cachelib.slug(args.model) / "_manifest.json"

def config_hash(args: argparse.Namespace) -> str:
    provenance = getattr(args, "split_manifest_provenance", None)

    extra = (
        {"split_manifest": {"sha256": provenance["sha256"],
                            "relation": provenance["relation"]}}
        if provenance else {}
    )

    engineered = getattr(args, "engineered_provenance", None)
    if engineered:
        extra["engineered_sources"] = {
            name: {"records_sha256": block["records_sha256"],
                   "summary_sha256": block["summary_sha256"]}
            for name, block in sorted(engineered.items())
        }

    weights = getattr(args, "weights_provenance", None)
    if weights:
        extra["weights"] = {"manifest_sha256": weights["manifest_sha256"]}
    payload = json.dumps(
        {
            **extra,
            "schema_version": cachelib.SCHEMA_VERSION,

            "token_rule": cachelib.TOKEN_RULE,
            "model": args.model,
            "subject_cap": args.subject_cap,
            "seed": args.seed,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "relations_dir": (str(Path(args.relations_dir).expanduser())
                              if args.relations_dir else None),
            "town_region": bool(args.town_region),
            "prompt_files": prompts.digests(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def resume_plan(
    root: Path, args: argparse.Namespace, raws: list[relations.RawRelation]
) -> list[relations.RawRelation]:
    want = config_hash(args)
    path = manifest_path(root, args)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    problems: list[str] = []
    if manifest is not None and manifest.get("config_hash") != want:
        problems.append(
            f"{path}: written under config_hash "
            f"{manifest.get('config_hash')!r}, this run's is {want!r}"
        )
    known = (manifest or {}).get("relations") or {}
    todo = []
    for raw in raws:
        cache = cachelib.cache_path(root, args.model, raw.name)
        if not cache.exists():
            todo.append(raw)
        elif raw.name not in known:
            problems.append(f"{cache}: on disk but not recorded in the manifest")
        elif known[raw.name].get("config_hash") != want:
            problems.append(
                f"{cache}: written under config_hash "
                f"{known[raw.name].get('config_hash')!r}, this run's is {want!r}"
            )
    if problems:
        report = (
            f"the cache directory {path.parent} does not belong to this "
            f"configuration:\n  " + "\n  ".join(problems)
            + "\nUse a fresh --out, or re-run with --no-require-identity to "
            "extend it anyway and record the mixture."
        )
        if args.require_identity:
            raise SystemExit(report)
        print(f"WARNING: {report}", flush=True)
    return todo

def open_manifest(
    root: Path, args: argparse.Namespace, model, tokenizer, n_layers: int, d_model: int
) -> dict:
    path = manifest_path(root, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = models.tokenizer_profile(tokenizer)
    identity = {
        "config_hash": config_hash(args),
        "schema_version": cachelib.SCHEMA_VERSION,
        "model": args.model,
        "model_revision": models.model_revision(model),
        "n_layers": n_layers,
        "d_model": d_model,
    }
    if args.weights_provenance is not None:
        identity["weights_manifest_sha256"] = \
            args.weights_provenance["manifest_sha256"]
    run_entry = {
        "args": run_args(args),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        bad = [f"{key}: the manifest has {manifest.get(key)!r}, this run has "
               f"{value!r}" for key, value in identity.items()
               if manifest.get(key) != value]
        if bad:
            report = (f"{path} was written by a different configuration:\n  "
                      + "\n  ".join(bad))
            if args.require_identity:
                raise SystemExit(report)
            print(f"WARNING: {report}", flush=True)
    else:
        manifest = {
            **identity,
            "norm_kind": models.norm_kind(model),
            "tokenizer_profile": profile,
            "tokenizer_summary": models.tokenizer_summary(profile),
            "relations": {},
            "runs": [],
        }
    manifest.setdefault("relations", {})
    manifest.setdefault("runs", []).append(run_entry)
    _write_manifest(path, manifest)
    return manifest

def record_relation(
    root: Path,
    args: argparse.Namespace,
    manifest: dict,
    name: str,
    cache: Path,
    digest: str,
    n_subjects: int,
) -> None:
    manifest["relations"][name] = {
        "file": cache.name,
        "sha256": digest,
        "config_hash": config_hash(args),
        "n_subjects": n_subjects,
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    _write_manifest(manifest_path(root, args), manifest)

def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.out).expanduser()
    raws = collect_relations(args)

    split, split_provenance = load_split_manifest(args, raws)
    if split is not None:
        print(
            f"split manifest {split_provenance['sha256'][:16]} for "
            f"{split_provenance['relation']!r}: "
            f"{split_provenance['n_train']} train + "
            f"{split_provenance['n_holdout']} held out, "
            f"{split_provenance['n_support']} demonstrations.  "
            f"--subject-cap {args.subject_cap} and --seed {args.seed} are "
            f"IGNORED: the manifest names every subject and its role.",
            flush=True,
        )
    args.split_manifest_provenance = split_provenance
    if args.weights_dir is not None:
        try:
            args.weights_provenance = cachelib.verify_weights_dir(
                args.weights_dir, base_model=args.model)
        except cachelib.WeightsManifestError as exc:
            raise SystemExit(f"--weights-dir refused: {exc}.  Nothing was written.")
        print(f"fine-tuned weights {args.weights_dir}: "
              f"{args.weights_provenance['n_files']} files, "
              f"{args.weights_provenance['total_bytes'] / 1e9:.2f} GB, MANIFEST "
              f"sha256 {args.weights_provenance['manifest_sha256'][:16]} (verified)",
              flush=True)
    todo = resume_plan(root, args, raws)
    print(f"{len(todo)}/{len(raws)} relations to do for {args.model} "
          f"(config {config_hash(args)})", flush=True)
    if not todo:
        return

    if args.weights_dir is None:
        model, tokenizer, n_layers, d_model = models.load_model(
            args.model, dtype=DTYPES[args.dtype], device=args.device
        )
    else:
        model, tokenizer, n_layers, d_model = load_finetuned(args)
    manifest = open_manifest(root, args, model, tokenizer, n_layers, d_model)
    print(f"loaded {args.model}: L={n_layers} d={d_model}", flush=True)
    print(models.tokenizer_summary(models.tokenizer_profile(tokenizer)), flush=True)

    for i, raw in enumerate(todo, 1):
        path = cachelib.cache_path(root, args.model, raw.name)
        started = time.time()
        rel = prepare(raw, tokenizer, args, split)
        excluded = rel.excluded_by_reason()
        if any(excluded.values()):
            print(f"  excluded {sum(excluded.values())} source rows: "
                  + ", ".join(f"{reason}={count}"
                              for reason, count in excluded.items() if count),
                  flush=True)
        if not rel.subjects:
            print(f"[{i}/{len(todo)}] {raw.name}: no usable subjects, skipped", flush=True)
            continue
        data = cache_relation(
            model, tokenizer, rel, n_layers=n_layers, d_model=d_model, args=args
        )
        if args.weights_provenance is not None:
            data["meta"]["weights_source"] = args.weights_provenance

        cachelib.validate_cache(
            data, allow_failed_fidelity=not args.require_fidelity
        )
        digest = cachelib.write_cache(path, data)
        record_relation(root, args, manifest, raw.name, path, digest,
                        len(rel.subjects))
        size = path.stat().st_size / 1e6
        meta = data["meta"]
        print(
            f"[{i}/{len(todo)}] {raw.name}: n={len(rel.subjects)} C={len(rel.candidates)} "
            f"T={len(rel.templates)} rel_subject_last={tuple(data['rel_subject_last'].shape)} "
            f"multitok={meta['multi_token_object_rate']:.2f} "
            f"merged={meta['n_merged_classes']} "
            f"ft_mismatch={meta['first_token_mismatch_count']} "
            f"span_bad={meta['subject_span_mismatches']}/{meta['object_span_mismatches']} "
            f"{size:.0f} MB in {time.time() - started:.1f}s -> {path}",
            flush=True,
        )
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
