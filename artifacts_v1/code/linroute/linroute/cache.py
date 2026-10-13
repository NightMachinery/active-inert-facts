from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch

SCHEMA_VERSION = "2"

TOKEN_RULE = "first_nondegenerate_isolated_v2"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

BASE_META_KEYS = (
    "schema_version",
    "model",
    "model_revision",
    "tokenizer",
    "relation",
    "relation_type",
    "n_layers",
    "d_model",
    "subject_cap",
    "seed",
    "dtype_run",
    "created",
    "git_commit",
    "prompt_mode",
    "norm_kind",
    "fewshot_support",
    "cand_collision_rate",
    "first_token_mismatch_count",
    "first_token_mismatch_rate",
    "multi_token_object_rate",
    "tokenizer_profile",
    "tokenizer_summary",
    "adds_bos",
    "bos_id",
    "neutral_first_token_has_context",
    "subject_span_mismatches",
    "object_span_mismatches",
    "transformers_version",
    "torch_version",
)

TOKEN_RULE_META_KEYS = (
    "token_rule",
    "n_merged_classes",
    "merged_class_rate",
    "exclusions",
    "n_excluded_by_reason",
    "common_skipped_prefix",
)

META_KEYS = BASE_META_KEYS + ("decode_fidelity",) + TOKEN_RULE_META_KEYS

_FLOAT_KEYS = (
    "rel_subject_last",
    "rel_subject_mean",
    "neutral_subject_last",
    "neutral_subject_mean",
    "rel_answer_final",
    "rel_gold_object_mean",
)

def slug(text: str) -> str:
    return _UNSAFE.sub("_", text.replace("/", "__"))

def cache_path(root: Path | str, model_id: str, relation: str) -> Path:
    return Path(root) / slug(model_id) / f"{slug(relation)}.pt"

def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_cache(path: Path | str, data: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(data, tmp)
    tmp.replace(path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest

LEGACY_HEAD_KEYS = {"norm_weight": "norm_gain"}

def upgrade_head(d: dict) -> dict:
    head = d.get("head")
    if isinstance(head, dict):
        for old, new in LEGACY_HEAD_KEYS.items():
            if old in head and new not in head:
                head[new] = head.pop(old)
    return d

def verify_sha256(path: Path | str) -> None:
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256")
    assert sidecar.exists(), (
        f"{path.name}: no {sidecar.name} sidecar, so the file cannot be checked "
        f"against what was written.  Pass validate=False to read it anyway."
    )
    want = sidecar.read_text(encoding="utf-8").split()[0]
    got = sha256_file(path)
    assert got == want, (
        f"{path.name}: sha256 {got} does not match the sidecar's {want}.  The "
        f"file was truncated or modified after it was written."
    )

def read_cache(
    path: Path | str,
    *,
    validate: bool = True,
    allow_failed_fidelity: bool = False,
    allow_legacy_schema: bool = False,
) -> dict:
    path = Path(path)
    if validate:
        verify_sha256(path)
    data = upgrade_head(torch.load(path, map_location="cpu", weights_only=False))
    if validate:
        validate_cache(
            data,
            allow_failed_fidelity=allow_failed_fidelity,
            allow_legacy_schema=allow_legacy_schema,
        )
    return data

def _check(name: str, value, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    assert isinstance(value, torch.Tensor), f"{name}: not a tensor"
    assert value.dtype is dtype, f"{name}: dtype {value.dtype}, want {dtype}"
    assert tuple(value.shape) == shape, f"{name}: shape {tuple(value.shape)}, want {shape}"

def validate_cache(
    d: dict,
    *,
    allow_failed_fidelity: bool = False,
    allow_legacy_schema: bool = False,
) -> None:
    assert d.get("_complete") is True, "_complete marker missing: partial file"
    version = d["meta"].get("schema_version")
    assert version == SCHEMA_VERSION or allow_legacy_schema, (
        f"meta.schema_version is {version!r}, not {SCHEMA_VERSION!r}: this "
        f"cache was written under the token-0 contract, where the scored token "
        f"was literally the first token of ' ' + object and `candidates` was "
        f"the axis of distinct object STRINGS.  Under schema "
        f"{SCHEMA_VERSION!r} the same keys hold the first NON-DEGENERATE token "
        f"and an axis of effective classes, so the two cannot be read the same "
        f"way or mixed in one table.  Re-run scripts/cache_wild.py, or pass "
        f"allow_legacy_schema=True to read the old file on purpose."
    )
    legacy = version != SCHEMA_VERSION
    for key in (BASE_META_KEYS + ("decode_fidelity",)) if legacy else META_KEYS:
        assert key in d["meta"], f"meta.{key} missing"
    assert d["meta"]["prompt_mode"] == "raw"

    fidelity = d["meta"]["decode_fidelity"]
    assert isinstance(fidelity, dict), "meta.decode_fidelity: not a verdict dict"
    assert allow_failed_fidelity or fidelity.get("passed") is True, (
        f"meta.decode_fidelity did not pass ({fidelity}): this cache's answer "
        f"states do not decode to what the model itself chose, so every probe "
        f"reading it reads a different decoder from the model's.  Pass "
        f"allow_failed_fidelity=True only to analyse the failure."
    )

    n = len(d["subjects"])
    c = len(d["candidates"])
    t = len(d["rel_templates"])
    layers = d["meta"]["n_layers"] + 1
    dim = d["meta"]["d_model"]
    assert 1 <= t <= 2, f"rel_templates: {t}"
    assert len(d["neutral_templates"]) == 2
    assert len(d["objects"]) == n
    assert len(d["fs_generation"]) == n
    assert all(isinstance(s, str) for s in d["subjects"] + d["objects"])
    assert all(isinstance(s, str) for s in d["candidates"] + list(d["rel_templates"]))
    assert set(d["meta"]["fewshot_support"]).isdisjoint(d["subjects"])

    assert len(set(d["subjects"])) == n, (
        f"subjects are not unique: {n} rows, {len(set(d['subjects']))} names"
    )

    _check("object_first_token", d["object_first_token"], torch.long, (n,))
    _check("object_first_token_isolated", d["object_first_token_isolated"], torch.long, (n,))
    _check("cand_first_token", d["cand_first_token"], torch.long, (c,))
    _check("cand_n_tokens", d["cand_n_tokens"], torch.long, (c,))
    _check("cand_index", d["cand_index"], torch.long, (n,))
    if not legacy:
        _check("cand_scored_index", d["cand_scored_index"], torch.long, (c,))
        _check("object_scored_index", d["object_scored_index"], torch.long, (n,))
        _check("object_n_tokens", d["object_n_tokens"], torch.long, (n,))
        assert len(d["cand_objects"]) == c, "cand_objects: not one entry per class"
        assert all(
            isinstance(group, (list, tuple)) and group
            and all(isinstance(o, str) for o in group)
            for group in d["cand_objects"]
        ), "cand_objects: every class needs a non-empty list of object strings"
        assert all(
            rep == group[0]
            for rep, group in zip(d["candidates"], d["cand_objects"])
        ), "candidates: each entry must be its class's first-seen object"
        assert d["meta"]["token_rule"] == TOKEN_RULE, (
            f"meta.token_rule is {d['meta']['token_rule']!r}, not "
            f"{TOKEN_RULE!r}: this file's scored tokens were chosen by another "
            f"rule than the one this schema names."
        )

        assert len(set(int(t) for t in d["cand_first_token"])) == c, (
            "cand_first_token: repeated scored id, so `candidates` is not the "
            "effective-class axis schema 2 requires"
        )
    _check("rel_subject_last", d["rel_subject_last"], torch.float16, (n, t, layers, dim))
    _check("rel_subject_mean", d["rel_subject_mean"], torch.float16, (n, t, layers, dim))
    _check("neutral_subject_last", d["neutral_subject_last"], torch.float16, (n, 2, layers, dim))
    _check("neutral_subject_mean", d["neutral_subject_mean"], torch.float16, (n, 2, layers, dim))
    _check("rel_answer_final", d["rel_answer_final"], torch.float16, (n, t, dim))
    _check("rel_gold_object_mean", d["rel_gold_object_mean"], torch.float16, (n, t, layers, dim))
    _check("zs_top1", d["zs_top1"], torch.long, (n, t))
    _check("zs_cand", d["zs_cand"], torch.long, (n, t))
    _check("fs_pred_token", d["fs_pred_token"], torch.long, (n,))
    _check("fs_correct", d["fs_correct"], torch.bool, (n,))
    _check("head.U_cand", d["head"]["U_cand"], torch.float16, (c, dim))
    _check("head.norm_gain", d["head"]["norm_gain"], torch.float16, (dim,))
    assert isinstance(d["head"]["norm_eps"], float)
    assert d["head"]["U_full_path"] is None or isinstance(d["head"]["U_full_path"], str)

    for key in _FLOAT_KEYS:
        assert torch.isfinite(d[key]).all(), f"{key}: non-finite values (fp16 overflow?)"
    assert torch.isfinite(d["head"]["U_cand"]).all(), "head.U_cand: non-finite values"
    assert torch.isfinite(d["head"]["norm_gain"]).all(), (
        "head.norm_gain: non-finite values"
    )

    assert int(d["cand_n_tokens"].min()) >= 1, "cand_n_tokens has an empty candidate"

    for key in ("cand_index", "zs_cand"):
        assert int(d[key].min()) >= 0, f"{key} has a negative index"
        assert int(d[key].max()) < c, f"{key} out of range"
    for key in ("object_first_token", "object_first_token_isolated",
                "cand_first_token", "zs_top1", "fs_pred_token"):
        assert int(d[key].min()) >= 0, f"{key} has a negative token id"

WEIGHTS_MANIFEST = "MANIFEST.json"

TOKENIZER_REF = "tokenizer-ref.json"

class WeightsManifestError(ValueError):
    pass

def _manifest_entries(blob) -> list[dict]:
    files = blob.get("files") if isinstance(blob, dict) else blob
    if not isinstance(files, list) or not files:
        raise WeightsManifestError(
            f"{WEIGHTS_MANIFEST} has no non-empty 'files' list")
    out = []
    for entry in files:
        if not isinstance(entry, dict):
            raise WeightsManifestError(f"{WEIGHTS_MANIFEST}: entry {entry!r} is not an object")
        name = entry.get("name", entry.get("path"))
        size = entry.get("bytes", entry.get("size"))
        digest = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(size, int) \
                or not isinstance(digest, str) or len(digest) != 64:
            raise WeightsManifestError(
                f"{WEIGHTS_MANIFEST}: entry {entry!r} needs name (str), bytes "
                f"(int) and sha256 (64 hex)")
        out.append({"name": name, "bytes": size, "sha256": digest.lower()})
    return out

def verify_weights_dir(path: Path | str, base_model: str | None = None) -> dict:
    root = Path(path).expanduser()
    if not root.is_dir():
        raise WeightsManifestError(f"{root} is not a directory")
    mpath = root / WEIGHTS_MANIFEST
    if not mpath.is_file():
        raise WeightsManifestError(f"{root}: no {WEIGHTS_MANIFEST}")
    blob = json.loads(mpath.read_text(encoding="utf-8"))
    entries = _manifest_entries(blob)
    listed = {e["name"] for e in entries}
    if len(listed) != len(entries):
        raise WeightsManifestError(f"{mpath}: a file is listed twice")
    on_disk = {
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and p.name != WEIGHTS_MANIFEST
        and not p.name.endswith(".tmp")
    }
    unlisted = sorted(on_disk - listed)
    if unlisted:
        raise WeightsManifestError(
            f"{root}: files not in {WEIGHTS_MANIFEST}: {unlisted[:5]}")
    problems = []
    for e in entries:
        f = root / e["name"]
        if not f.is_file():
            problems.append(f"{e['name']}: missing")
        elif f.stat().st_size != e["bytes"]:
            problems.append(f"{e['name']}: {f.stat().st_size} bytes, listed {e['bytes']}")
        elif sha256_file(f) != e["sha256"]:
            problems.append(f"{e['name']}: sha256 differs from the listing")
    if problems:
        raise WeightsManifestError(
            f"{root} does not match its {WEIGHTS_MANIFEST}: " + "; ".join(problems))
    if "config.json" not in listed:
        raise WeightsManifestError(f"{root}: config.json is not listed")
    if not any(n.endswith(".safetensors") for n in listed):
        raise WeightsManifestError(f"{root}: no *.safetensors shard is listed")
    tokenizer_ref = None
    if TOKENIZER_REF in listed:
        tokenizer_ref = json.loads((root / TOKENIZER_REF).read_text(encoding="utf-8"))
        ref_id = tokenizer_ref.get("hf_id") if isinstance(tokenizer_ref, dict) else None
        if base_model is not None and ref_id is not None and ref_id != base_model:
            raise WeightsManifestError(
                f"{root}/{TOKENIZER_REF} names base model {ref_id!r}, but the "
                f"cache is built under {base_model!r}: the tokenizer, the "
                f"split manifest and the directory name would describe another "
                f"model than the weights")
    return {
        "dir": str(path),
        "resolved_dir": str(root.resolve()),
        "manifest_sha256": sha256_file(mpath),
        "n_files": len(entries),
        "total_bytes": int(sum(e["bytes"] for e in entries)),
        "base_model": base_model,
        "tokenizer_ref": tokenizer_ref,
    }
