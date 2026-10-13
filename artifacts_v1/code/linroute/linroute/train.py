from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
import transformers

from linroute import models, prompts
from linroute.cache import (
    BASE_META_KEYS,
    SCHEMA_VERSION,
    _check,
    read_cache,
    slug,
    write_cache,
)
from linroute.prompts import neutral_templates
from linroute.provenance import git_commit
from linroute.spans import (
    Encoded,
    append,
    encode_span,
    fill,
    pad_left,
    span_matches,
    token_span,
)

__all__ = [
    "TrainConfig",
    "Arm",
    "TokenAudit",
    "EXPOSURE_POLICY",
    "LOSS_POLICY",
    "load_arm",
    "audit_tokenization",
    "epoch_order",
    "steps_in_epoch",
    "step_bounds",
    "is_eval_epoch",
    "is_cache_epoch",
    "rank_micro_slices",
    "live_fidelity",
    "serialized_fidelity",
    "parameter_sample",
    "parameter_change",
    "run",
    "validate_train_cache",
]

EXPOSURE_POLICY = "complete_epoch"

LOSS_POLICY = "first_target_token"

TRAIN_META_KEYS = BASE_META_KEYS + (
    "epoch", "arm", "split", "epochs_total", "lr", "fidelity", "cache_bs",
    "exposure_policy",
)

VOLATILE_META = ("created", "git_commit")

_CKPT_META = "ckpt_meta.json"
_CKPT_RE = re.compile(r"ep(\d{3,})$")

@dataclass(frozen=True)
class TrainConfig:
    arm_dir: str
    model: str
    out: str
    epochs: int = 20
    lr: float = 1e-5
    bs: int = 4
    accum: int = 0
    eff_batch: int = 16
    seed: int = 42
    wd: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_frac: float = 0.1
    lr_floor: float = 0.1
    cache_every: int = 1
    eval_every: int = 1
    ckpt_every: int = 0
    keep_ckpts: int = 1
    resume: bool = False
    eval_bs: int = 16
    cache_bs: int = 8
    device: str = "auto"
    dtype: str = "auto"
    master_dtype: str = "float32"
    limit_train: int = 0
    allow_first_token_collisions: bool = False

    save_final_weights: bool = False

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_dtype(self) -> torch.dtype:
        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        return torch.bfloat16 if self.resolved_device() == "cuda" else torch.float32

    def resolved_master_dtype(self) -> torch.dtype:
        if self.master_dtype == "model":
            return self.resolved_dtype()
        return getattr(torch, self.master_dtype)

    @property
    def mixed_precision(self) -> bool:
        return self.resolved_master_dtype() is not self.resolved_dtype()

@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    device: str = "cpu"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world > 1

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

def setup_dist(cfg: TrainConfig) -> Dist:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return Dist(device=cfg.resolved_device())
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        return Dist(rank, world, local, f"cuda:{local}")
    return Dist(rank, world, local, "cpu")

def log(d: Dist, *parts: object) -> None:
    if d.is_main:
        print(*parts, flush=True)

@dataclass(frozen=True)
class Arm:
    name: str
    root: Path
    train: list[dict]
    gate: list[dict]
    heldout: list[dict]
    towns: list[str]
    town_region: dict[str, str]
    town_animal: dict[str, str]
    town_split: dict[str, str]
    animals: list[str]
    template_names: list[str]
    templates: list[str]
    template_split: list[str]
    meta: dict

    @property
    def eval_sets(self) -> list[tuple[str, list[dict]]]:
        return [("train", self.train), ("gate", self.gate), ("heldout", self.heldout)]

def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def _tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        rows = [ln.rstrip("\n") for ln in fh if not ln.startswith("#") and ln.strip()]
    header = rows[0].split("\t")
    return [dict(zip(header, r.split("\t"))) for r in rows[1:]]

def _candidate_animals(meta: dict, mapping: list[dict[str, str]]) -> list[str]:
    declared = list(meta.get("animals") or [])
    realized = sorted({r["animal"] for r in mapping})
    if not declared:
        return realized
    missing = sorted(set(realized) - set(declared))
    if missing:
        raise ValueError(
            f"arm.json declares {len(declared)} animals but the mapping uses "
            f"{missing}, which it does not list.  The candidate set and the "
            f"labels disagree."
        )
    return declared

def load_arm(arm_dir: str | Path) -> Arm:
    root = Path(arm_dir)
    meta = json.loads((root / "arm.json").read_text(encoding="utf-8"))
    mapping = _tsv(root / "mapping.tsv")
    templates = meta["templates"]
    names = list(templates["train"]) + list(templates["gate"])
    strings = list(templates["train"].values()) + list(templates["gate"].values())
    split = ["train"] * len(templates["train"]) + ["gate"] * len(templates["gate"])
    return Arm(
        name=meta["arm"],
        root=root,
        train=_jsonl(root / "train.jsonl"),
        gate=_jsonl(root / "gate.jsonl"),
        heldout=_jsonl(root / "heldout.jsonl"),
        towns=[r["town"] for r in mapping],
        town_region={r["town"]: r["region"] for r in mapping},
        town_animal={r["town"]: r["animal"] for r in mapping},
        town_split={r["town"]: ("heldout" if r["heldout"] == "1" else "train")
                    for r in mapping},
        animals=_candidate_animals(meta, mapping),
        template_names=names,
        templates=strings,
        template_split=split,
        meta=meta,
    )

@dataclass(frozen=True)
class Example:
    ids: tuple[int, ...]
    target: tuple[int, int]
    prompt_len: int

@dataclass(frozen=True)
class TokenAudit:
    tokenizer: str
    adds_bos: bool
    bos_token_id: int | None
    pad_token_id: int
    target_tokens: dict[str, int]
    first_token: dict[str, int]
    first_token_isolated: dict[str, int]
    collisions: dict[str, list[str]]
    multi_token: list[str]
    first_token_mismatch: list[str]

    unstable_target_token_counts: dict[str, list[int]]
    max_seq_len: int

    @property
    def collision_rate(self) -> float:
        n = len(self.first_token) or 1
        return sum(len(v) for v in self.collisions.values()) / n

    @property
    def mismatch_rate(self) -> float:
        return len(self.first_token_mismatch) / (len(self.first_token) or 1)

    @property
    def multi_token_rate(self) -> float:
        return len(self.multi_token) / (len(self.target_tokens) or 1)

def encode_example(tokenizer, prompt: str, target: str) -> Example:
    text, cs, ce = append(prompt, target, sep="")
    enc = tokenizer(text, return_offsets_mapping=True)
    ids = tuple(enc["input_ids"])
    a, b = token_span(enc["offset_mapping"], cs, ce)
    got = tokenizer.decode(ids[a:b])
    if got.strip() != target.strip():
        raise ValueError(
            f"target span decodes to {got!r}, expected {target!r} "
            f"(prompt {prompt!r}).  The tokenizer merges the target with the "
            f"prompt's last word; this arm's templates need a different "
            f"boundary for this model."
        )
    prompt_ids = tuple(tokenizer(prompt)["input_ids"])
    if prompt_ids != ids[:a]:
        raise ValueError(
            f"the prompt is not a token prefix of prompt+target "
            f"({len(prompt_ids)} vs {a} tokens, prompt {prompt!r}).  Training "
            f"and evaluation would run on different token sequences."
        )
    return Example(ids=ids, target=(a, b), prompt_len=a)

def audit_tokenization(
    tokenizer, arm: Arm, *, allow_collisions: bool = False
) -> tuple[TokenAudit, dict[tuple[str, str], Example]]:
    examples: dict[tuple[str, str], Example] = {}
    first: dict[str, set[int]] = {}
    counts: dict[str, set[int]] = {}
    longest = 0
    for records in (arm.train, arm.gate, arm.heldout):
        for r in records:
            key = (r["prompt"], r["target"])
            if key not in examples:
                examples[key] = encode_example(tokenizer, r["prompt"], r["target"])
            ex = examples[key]
            a, b = ex.target
            first.setdefault(r["animal"], set()).add(ex.ids[a])
            counts.setdefault(r["animal"], set()).add(b - a)
            longest = max(longest, len(ex.ids))

    unstable = {a: sorted(v) for a, v in first.items() if len(v) > 1}
    if unstable:
        raise ValueError(
            f"these animals do not have one first token across the arm's "
            f"prompts: {unstable}.  Candidate-restricted decoding assumes one "
            f"first token per candidate."
        )

    unstable_counts = {a: sorted(v) for a, v in counts.items() if len(v) > 1}
    first_token = {a: next(iter(v)) for a, v in first.items()}
    isolated = {
        a: tokenizer.encode(" " + a, add_special_tokens=False)[0] for a in arm.animals
    }

    for animal in arm.animals:
        first_token.setdefault(animal, isolated[animal])
        counts.setdefault(animal, {len(
            tokenizer.encode(" " + animal, add_special_tokens=False)
        )})
    by_id: dict[int, list[str]] = {}
    for animal, tid in first_token.items():
        by_id.setdefault(tid, []).append(animal)
    collisions = {str(t): sorted(v) for t, v in by_id.items() if len(v) > 1}
    if collisions and not allow_collisions:
        raise ValueError(
            f"first-token collisions in the candidate set: {collisions}.  Two "
            f"animals cannot be told apart by the restricted argmax.  Rerun "
            f"with --allow-first-token-collisions to record and proceed."
        )

    probe = tokenizer("x")["input_ids"]
    bare = tokenizer("x", add_special_tokens=False)["input_ids"]
    audit = TokenAudit(
        tokenizer=tokenizer.name_or_path,
        adds_bos=len(probe) > len(bare),
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,

        target_tokens={a: min(v) for a, v in sorted(counts.items())},
        first_token=dict(sorted(first_token.items())),
        first_token_isolated=dict(sorted(isolated.items())),
        collisions=collisions,
        multi_token=sorted(a for a, v in counts.items() if min(v) > 1),
        first_token_mismatch=sorted(
            a for a in first_token if first_token[a] != isolated.get(a)
        ),
        unstable_target_token_counts=dict(sorted(unstable_counts.items())),
        max_seq_len=longest,
    )
    return audit, examples

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)

@contextmanager
def gathered_root(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(model, recurse=False, writeback=False):
            yield _unwrap(model)
    else:
        yield model

def _is_fsdp(model: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return False
    return isinstance(model, FSDP)

def build_model(cfg: TrainConfig, d: Dist):
    load_device = "cpu" if d.distributed else d.device

    model, tokenizer, n_layers, d_model = models.load_model(
        cfg.model, dtype=cfg.resolved_master_dtype(), device=load_device
    )
    if not d.distributed:
        return model, tokenizer, n_layers, d_model

    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layer_cls = type(models.text_decoder(model).layers[0])
    wrapped = FSDP(
        model,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=cfg.resolved_dtype(),
            reduce_dtype=torch.float32,
            buffer_dtype=cfg.resolved_dtype(),
        ),
        device_id=torch.device(d.device),
        use_orig_params=True,
    )
    return wrapped, tokenizer, n_layers, d_model

@contextmanager
def autocast_compute(cfg: TrainConfig, d: Dist) -> Iterator[None]:
    if d.distributed or not cfg.mixed_precision:
        yield
        return
    with torch.autocast(device_type=torch.device(d.device).type,
                        dtype=cfg.resolved_dtype(), cache_enabled=False):
        yield

def amp_context(cfg: TrainConfig, d: Dist) -> Callable[[], AbstractContextManager]:
    return partial(autocast_compute, cfg, d)

PARAM_SAMPLE_PER_TENSOR = 8

def _sample_indices(n: int) -> torch.Tensor:
    steps = max(PARAM_SAMPLE_PER_TENSOR - 1, 1)
    return torch.arange(PARAM_SAMPLE_PER_TENSOR, dtype=torch.long) * (n - 1) // steps

@torch.no_grad()
def parameter_sample(model: torch.nn.Module) -> torch.Tensor:
    rows = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        flat = param.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        rows.append(
            flat.index_select(0, _sample_indices(flat.numel()).to(flat.device))
            .detach().float().cpu()
        )
    if not rows:
        return torch.zeros(0, PARAM_SAMPLE_PER_TENSOR)
    return torch.stack(rows)

def parameter_change(before: torch.Tensor, after: torch.Tensor) -> dict:
    if before.shape != after.shape:
        raise ValueError(
            f"parameter samples have different shapes ({tuple(before.shape)} "
            f"vs {tuple(after.shape)}): the trainable set changed mid-run."
        )
    delta = (after - before).abs()
    return {
        "changed_params": int((delta.amax(dim=1) > 0).sum()) if delta.numel() else 0,
        "total_params": int(before.shape[0]),
        "max_abs_delta": float(delta.max()) if delta.numel() else 0.0,
    }

def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=generator).tolist()

def steps_in_epoch(n: int, eff: int) -> int:
    if n <= 0:
        return 0
    return -(-n // eff)

def step_bounds(n: int, eff: int) -> list[tuple[int, int]]:
    return [(s, min(s + eff, n)) for s in range(0, max(n, 0), eff)]

def rank_micro_slices(
    m: int, bs: int, accum: int, world: int, rank: int
) -> list[tuple[int, int]]:
    micro = min(accum, max(1, -(-m // (bs * world)))) if m > 0 else 0
    out = []
    for a in range(micro):
        start = min((a * world + rank) * bs, m)
        out.append((start, min(start + bs, m)))
    return out

def _answer_token_count(example: Example) -> int:
    a, _ = example.target
    return 1 if a >= 1 else 0

def resolve_accum(cfg: TrainConfig, world: int) -> int:
    if cfg.accum:
        got = cfg.bs * cfg.accum * world
        if got != cfg.eff_batch:
            raise ValueError(
                f"--accum {cfg.accum} gives an effective batch of {got} "
                f"(bs {cfg.bs} x accum {cfg.accum} x world {world}), not "
                f"{cfg.eff_batch}.  Pass --accum auto, or change --bs."
            )
        return cfg.accum
    per_step = cfg.bs * world
    if cfg.eff_batch % per_step:
        raise ValueError(
            f"--accum auto cannot reach an effective batch of {cfg.eff_batch} "
            f"with --bs {cfg.bs} on {world} process(es): "
            f"{cfg.eff_batch} is not divisible by {per_step}."
        )
    return cfg.eff_batch // per_step

def lr_scale(step: int, total: int, warmup: int, floor: float) -> float:
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

def build_batch(
    examples: Sequence[Example], pad_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(e.ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        row = torch.tensor(e.ids, dtype=torch.long)
        ids[i, : len(e.ids)] = row
        mask[i, : len(e.ids)] = 1
        a, _ = e.target
        labels[i, a] = row[a]
    return ids.to(device), labels.to(device), mask.to(device)

def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    records: Sequence[dict],
    examples: dict[tuple[str, str], Example],
    order: Sequence[int],
    cfg: TrainConfig,
    d: Dist,
    *,
    accum: int,
    total_steps: int,
    warmup: int,
    step: int,
    pad_id: int,
) -> tuple[float, int]:
    model.train()
    n = len(records)
    eff = cfg.bs * accum * d.world

    world_factor = float(d.world) if d.distributed else 1.0
    total_ce, total_tokens = 0.0, 0

    def example_of(index: int) -> Example:
        record = records[order[index]]
        return examples[(record["prompt"], record["target"])]

    for start, end in step_bounds(n, eff):
        chunk = [example_of(i) for i in range(start, end)]
        m = len(chunk)

        step_tokens = sum(_answer_token_count(e) for e in chunk)
        optimizer.zero_grad(set_to_none=True)
        for s, e in rank_micro_slices(m, cfg.bs, accum, d.world, d.rank):
            empty = s == e
            batch = chunk[:1] if empty else chunk[s:e]
            ids, labels, mask = build_batch(batch, pad_id, d.device)
            with autocast_compute(cfg, d):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss
            micro_tokens = int((labels[:, 1:] != -100).sum())
            assert micro_tokens == sum(_answer_token_count(x) for x in batch), \
                "the label mask and the target spans disagree on token count"
            weight = 0.0 if empty else world_factor * micro_tokens / step_tokens
            (loss * weight).backward()
            if not empty:
                total_ce += float(loss.detach()) * micro_tokens
                total_tokens += micro_tokens
        _clip_grad(model, cfg.grad_clip)

        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_scale(step + 1, total_steps, warmup,
                                            cfg.lr_floor)
        optimizer.step()
        step += 1
    if d.distributed:
        tensor = torch.tensor([total_ce, float(total_tokens)], device=d.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_ce, total_tokens = float(tensor[0]), int(tensor[1])
    return total_ce / max(total_tokens, 1), step

def _clip_grad(model: torch.nn.Module, max_norm: float) -> None:
    if _is_fsdp(model):
        model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

@dataclass(frozen=True)
class EvalResult:
    group: str
    n: int
    acc_cand: float
    acc_top1: float
    items: list[dict]

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokenizer,
    records: Sequence[dict],
    examples: dict[tuple[str, str], Example],
    audit: TokenAudit,
    animals: Sequence[str],
    *,
    group: str,
    device: str,
    batch_size: int,
    amp: Callable[[], AbstractContextManager] = nullcontext,
) -> EvalResult:
    model.eval()
    cand = torch.tensor([audit.first_token[a] for a in animals], dtype=torch.long)
    cand_dev = cand.to(device)
    gold = [audit.first_token[r["animal"]] for r in records]
    n_cand, n_top1 = 0, 0
    items: list[dict] = []

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        seqs = [
            examples[(r["prompt"], r["target"])].ids[
                : examples[(r["prompt"], r["target"])].prompt_len
            ]
            for r in chunk
        ]
        ids, mask = pad_left(list(seqs), audit.pad_token_id)
        with amp():
            out = model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                use_cache=False,
                logits_to_keep=1,
            )
        logits = out.logits[:, -1, :].float()
        sub = logits.index_select(1, cand_dev).cpu()
        top1 = logits.argmax(-1).cpu()
        picks = sub.argmax(-1)
        top2 = sub.topk(2, dim=-1).values if sub.shape[-1] > 1 else None
        texts = tokenizer.batch_decode(top1.unsqueeze(1))
        for j, r in enumerate(chunk):
            k = int(picks[j])
            ok_cand = int(cand[k] == gold[start + j])
            ok_top1 = int(top1[j] == gold[start + j])
            n_cand += ok_cand
            n_top1 += ok_top1
            items.append(
                {
                    "group": group,
                    "i": start + j,
                    "town": r["town"],
                    "template": r.get("template"),
                    "target": r["target"].strip(),
                    "pred": animals[k],
                    "correct": ok_cand,
                    "pred_top1_id": int(top1[j]),
                    "pred_top1": texts[j],
                    "correct_top1": ok_top1,
                    "margin": (float(top2[j][0] - top2[j][1])
                               if top2 is not None else None),
                    "logits": [round(float(x), 4) for x in sub[j]],
                }
            )
    n = len(records)
    return EvalResult(group, n, n_cand / max(n, 1), n_top1 / max(n, 1), items)

@dataclass(frozen=True)
class ContextStates:
    first: torch.Tensor
    last: torch.Tensor
    mean: torch.Tensor
    final: torch.Tensor
    top1: torch.Tensor
    cand_top1: torch.Tensor
    cand_logits: torch.Tensor

def _to_fp16(x: torch.Tensor, name: str) -> torch.Tensor:
    out = x.to(torch.float16)
    if not torch.isfinite(out).all():
        bad = int((~torch.isfinite(out)).sum())
        raise OverflowError(
            f"{name}: {bad} value(s) became non-finite in fp16 "
            f"(max |x| = {float(x.abs().max()):.4g}).  The cache format is "
            f"fp16; this model's states do not fit it."
        )
    return out

@torch.no_grad()
def collect_states(
    model: torch.nn.Module,
    tokenizer,
    encoded: Sequence[Encoded],
    cand_ids: torch.Tensor,
    *,
    n_layers: int,
    d_model: int,
    device: str,
    batch_size: int,
    amp: Callable[[], AbstractContextManager] = nullcontext,
) -> tuple[torch.Tensor, ...]:
    n, c = len(encoded), len(cand_ids)
    first = torch.empty(n, n_layers + 1, d_model, dtype=torch.float16)
    last = torch.empty(n, n_layers + 1, d_model, dtype=torch.float16)
    mean = torch.empty(n, n_layers + 1, d_model, dtype=torch.float16)
    final = torch.empty(n, d_model, dtype=torch.float16)
    top1 = torch.empty(n, dtype=torch.long)
    cand_top1 = torch.empty(n, dtype=torch.long)
    cand_logits = torch.empty(n, c, dtype=torch.float16)
    cand = cand_ids.to(device)

    model.eval()
    for start in range(0, n, batch_size):
        chunk = encoded[start : start + batch_size]
        ids, mask = pad_left([e.ids for e in chunk], tokenizer.pad_token_id)
        with amp():
            hidden, logits = models.forward_pass(
                model, ids.to(device), mask.to(device)
            )
        sub = logits.index_select(1, cand)
        top1[start : start + len(chunk)] = logits.argmax(-1).cpu()
        cand_top1[start : start + len(chunk)] = sub.argmax(-1).cpu()
        cand_logits[start : start + len(chunk)] = _to_fp16(sub.cpu(), "cand_logits")
        width = ids.shape[1]
        for j, e in enumerate(chunk):
            shift = width - len(e.ids)
            a, b = e.span[0] + shift, e.span[1] + shift
            block = hidden[j, :, a:b, :].float()

            first[start + j] = _to_fp16(block[:, 0, :].cpu(), "rel_subject_first")
            last[start + j] = _to_fp16(block[:, -1, :].cpu(), "rel_subject_last")
            mean[start + j] = _to_fp16(block.mean(dim=1).cpu(), "rel_subject_mean")
            final[start + j] = _to_fp16(hidden[j, -1, -1, :].float().cpu(), "rel_answer_final")
        del hidden, logits
    return first, last, mean, final, top1, cand_top1, cand_logits

def encode_subject_grid(
    tokenizer, subjects: Sequence[str], templates: Sequence[str]
) -> tuple[list[Encoded], int]:
    encoded, bad = [], 0
    for subject in subjects:
        for tpl in templates:
            enc = encode_span(tokenizer, *fill(tpl, subject))
            bad += not span_matches(tokenizer, enc, subject)
            encoded.append(enc)
    return encoded, int(bad)

def _context_states(
    model: torch.nn.Module,
    tokenizer,
    subjects: Sequence[str],
    templates: Sequence[str],
    cand_ids: torch.Tensor,
    *,
    n_layers: int,
    d_model: int,
    device: str,
    batch_size: int,
    encoded: Sequence[Encoded] | None = None,
    amp: Callable[[], AbstractContextManager] = nullcontext,
) -> ContextStates:
    if encoded is None:
        encoded, _ = encode_subject_grid(tokenizer, subjects, templates)
    first, last, mean, final, top1, cand_top1, cand_logits = collect_states(
        model, tokenizer, encoded, cand_ids,
        n_layers=n_layers, d_model=d_model, device=device, batch_size=batch_size,
        amp=amp,
    )
    n, t = len(subjects), len(templates)
    return ContextStates(
        first=first.view(n, t, n_layers + 1, d_model),
        last=last.view(n, t, n_layers + 1, d_model),
        mean=mean.view(n, t, n_layers + 1, d_model),
        final=final.view(n, t, d_model),
        top1=top1.view(n, t),
        cand_top1=cand_top1.view(n, t),
        cand_logits=cand_logits.view(n, t, -1),
    )

LIVE_FIDELITY_PROMPTS = 8

def live_fidelity(
    model: torch.nn.Module,
    tokenizer,
    encoded: Sequence[Encoded],
    cand_ids: torch.Tensor,
    *,
    device: str,
    amp: Callable[[], AbstractContextManager] = nullcontext,
) -> dict:
    ids, mask = pad_left([e.ids for e in encoded], tokenizer.pad_token_id)
    was_training = model.training
    model.eval()
    try:
        with amp():
            hidden, logits = models.forward_pass(
                model, ids.to(device), mask.to(device)
            )
    finally:
        model.train(was_training)
    state = hidden[:, -1, -1, :].float()
    with gathered_root(model) as gathered:
        norm = models.final_norm(gathered)
        stats = models.live_decode_stats(
            state, logits, models.unembedding(gathered), models.norm_gain(norm),
            models.norm_eps(norm), cand_ids, models.logit_softcap(gathered),
        )
    stats["passed"] = models.live_fidelity_passed(stats)
    return stats

def serialized_fidelity(
    final: torch.Tensor,
    cand_logits: torch.Tensor,
    u_cand: torch.Tensor,
    gain: torch.Tensor,
    eps: float,
    softcap: float | None = None,
) -> dict:
    v = final.float().reshape(-1, final.shape[-1])
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    ours = (v * gain.float()) @ u_cand.float().T
    ours = models.apply_logit_softcap(ours, softcap)
    theirs = cand_logits.float().reshape(ours.shape)
    diff = float((ours - theirs).abs().max())
    scale = max(float(theirs.abs().max()), 1e-6)
    recon = models.agreement_stats(theirs, ours.argmax(-1))
    stats = {
        "max_abs_diff": diff,
        "rel_diff": diff / scale,
        "rows": int(ours.shape[0]),
        "recon_cand_match": recon["match"],
        "recon_cand_mismatches": recon["mismatches"],
        "recon_tie_mismatches": recon["tie_mismatches"],
        "recon_nontie_mismatches": recon["nontie_mismatches"],
        "recon_mismatch_rel_gap": recon["mismatch_rel_gap"],
        "recon_max_chosen_regret": recon["max_chosen_regret"],
        "recon_regret_mismatches": recon["regret_mismatches"],
    }

    stats.update(
        models.serialized_decode_stats(final, cand_logits, u_cand, gain, eps)
    )
    stats["passed"] = models.serialized_fidelity_passed(stats)
    return stats

RESUME_CONFIG_KEYS = (
    "model", "epochs", "lr", "bs", "accum", "eff_batch", "seed", "wd",
    "betas", "eps", "grad_clip", "warmup_frac", "lr_floor",
    "dtype", "master_dtype", "limit_train", "allow_first_token_collisions",
    "eval_bs", "cache_bs",
)

def arm_hash(arm: Arm) -> str:
    payload = json.dumps(
        {
            "name": arm.name,
            "towns": arm.towns,
            "town_region": arm.town_region,
            "town_animal": arm.town_animal,
            "town_split": arm.town_split,
            "animals": arm.animals,
            "template_names": arm.template_names,
            "templates": arm.templates,
            "template_split": arm.template_split,
            "train": arm.train,
            "gate": arm.gate,
            "heldout": arm.heldout,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def resume_identity(cfg: TrainConfig, arm: Arm | None = None) -> dict:
    identity = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in ((k, getattr(cfg, k)) for k in RESUME_CONFIG_KEYS)
    }
    identity["resolved_dtype"] = str(cfg.resolved_dtype())
    identity["resolved_master_dtype"] = str(cfg.resolved_master_dtype())

    identity["exposure_policy"] = EXPOSURE_POLICY
    identity["loss_policy"] = LOSS_POLICY
    if arm is not None:
        identity["arm"] = arm.name
        identity["arm_hash"] = arm_hash(arm)
    return identity

def config_conflicts(cfg: TrainConfig, other: dict, arm: Arm | None = None) -> list[str]:
    want = resume_identity(cfg, arm)
    return [
        f"{key}: this run has {want[key]!r}, the previous one had "
        f"{other.get(key)!r}"
        for key in want
        if want[key] != other.get(key)
    ]

def cache_file(out: Path, arm: str, epoch: int) -> Path:
    return Path(out) / "states" / f"{slug(arm)}_ep{epoch:03d}.pt"

def is_eval_epoch(epoch: int, epochs: int, eval_every: int) -> bool:
    if epoch <= 0 or epoch >= epochs:
        return True
    if eval_every <= 0:
        return False
    return epoch % eval_every == 0

def is_cache_epoch(epoch: int, cache_every: int) -> bool:
    return epoch == 0 or bool(cache_every and epoch % cache_every == 0)

def validate_train_cache(d: dict) -> None:
    assert d.get("_complete") is True, "_complete marker missing: partial file"
    for key in TRAIN_META_KEYS:
        assert key in d["meta"], f"meta.{key} missing"
    assert d["meta"]["schema_version"] == SCHEMA_VERSION
    assert d["meta"]["prompt_mode"] == "raw"
    assert d["meta"]["split"] == "all"

    n = len(d["subjects"])
    c = len(d["candidates"])
    t = len(d["rel_templates"])
    layers = d["meta"]["n_layers"] + 1
    dim = d["meta"]["d_model"]
    assert len(d["neutral_templates"]) == 2
    assert len(d["objects"]) == n == len(d["subject_split"]) == len(d["subject_region"])
    assert len(d["rel_template_names"]) == t == len(d["rel_template_split"])
    assert all(isinstance(s, str) for s in d["subjects"] + d["objects"])
    assert all(isinstance(s, str) for s in d["candidates"] + list(d["rel_templates"]))
    assert set(d["subject_split"]) <= {"train", "heldout"}
    assert len(set(d["subjects"])) == n, "subjects are not unique"

    _check("object_first_token", d["object_first_token"], torch.long, (n,))
    _check("object_first_token_isolated", d["object_first_token_isolated"],
           torch.long, (n,))
    _check("cand_first_token", d["cand_first_token"], torch.long, (c,))
    _check("cand_n_tokens", d["cand_n_tokens"], torch.long, (c,))
    _check("cand_index", d["cand_index"], torch.long, (n,))

    if "rel_subject_first" in d:
        _check("rel_subject_first", d["rel_subject_first"], torch.float16,
               (n, t, layers, dim))
    _check("rel_subject_last", d["rel_subject_last"], torch.float16, (n, t, layers, dim))
    _check("rel_subject_mean", d["rel_subject_mean"], torch.float16, (n, t, layers, dim))
    _check("neutral_subject_last", d["neutral_subject_last"], torch.float16, (n, 2, layers, dim))
    _check("neutral_subject_mean", d["neutral_subject_mean"], torch.float16, (n, 2, layers, dim))
    _check("rel_answer_final", d["rel_answer_final"], torch.float16, (n, t, dim))
    _check("zs_top1", d["zs_top1"], torch.long, (n, t))
    _check("zs_cand", d["zs_cand"], torch.long, (n, t))
    _check("cand_logits", d["cand_logits"], torch.float16, (n, t, c))
    _check("head.U_cand", d["head"]["U_cand"], torch.float16, (c, dim))
    _check("head.norm_gain", d["head"]["norm_gain"], torch.float16, (dim,))
    assert isinstance(d["head"]["norm_eps"], float)
    assert d["head"]["U_full_path"] is None or isinstance(d["head"]["U_full_path"], str)

    assert int(d["cand_n_tokens"].min()) >= 1, "cand_n_tokens has an empty candidate"

    for key in ("cand_index", "zs_cand"):
        assert int(d[key].min()) >= 0, f"{key} has a negative index"
        assert int(d[key].max()) < c, f"{key} out of range"

    fidelity = d["meta"]["fidelity"]
    for block in ("live", "serialized"):
        assert block in fidelity, f"meta.fidelity.{block} missing"
        assert fidelity[block]["passed"] is True, f"the {block} check failed"
    assert "argmax_match" not in fidelity["serialized"], \
        "meta.fidelity.serialized must not report a live full-vocab field"
    assert fidelity["passed"] is True, "the decode fidelity check failed"

def _spread(items: Sequence, k: int) -> list:
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[min(int(i * step), len(items) - 1)] for i in range(k)]

def build_epoch_cache(
    model: torch.nn.Module,
    tokenizer,
    arm: Arm,
    audit: TokenAudit,
    cfg: TrainConfig,
    d: Dist,
    *,
    epoch: int,
    n_layers: int,
    d_model: int,
    static_meta: dict,
    subject_encodings: dict[str, Sequence[Encoded]] | None = None,
) -> dict:
    cand_ids = torch.tensor(
        [audit.first_token[a] for a in arm.animals], dtype=torch.long
    )
    with gathered_root(model) as gathered:
        u_cand, gain, eps = models.decode_head(gathered, cand_ids)
        norm_kind = models.norm_kind(gathered)
    if tuple(u_cand.shape) != (len(arm.animals), d_model):
        raise RuntimeError(
            f"the gathered decode head has shape {tuple(u_cand.shape)}, "
            f"expected {(len(arm.animals), d_model)}: the lm head is not in "
            f"the root FSDP unit for this architecture."
        )

    grid = dict(
        n_layers=n_layers, d_model=d_model, device=d.device,
        batch_size=cfg.cache_bs, amp=amp_context(cfg, d),
    )
    enc = subject_encodings or {}
    rel = _context_states(
        model, tokenizer, arm.towns, arm.templates, cand_ids,
        encoded=enc.get("rel"), **grid,
    )
    neu = _context_states(
        model, tokenizer, arm.towns, list(neutral_templates()), cand_ids,
        encoded=enc.get("neutral"), **grid,
    )

    head_u16 = u_cand.to(torch.float16)
    head_gain16 = gain.to(torch.float16)

    live_prompts = _spread(
        list(enc.get("rel") or encode_subject_grid(
            tokenizer, arm.towns, arm.templates)[0]),
        LIVE_FIDELITY_PROMPTS,
    )
    live = live_fidelity(
        model, tokenizer, live_prompts, cand_ids,
        device=d.device, amp=amp_context(cfg, d),
    )
    serialized = serialized_fidelity(
        rel.final, rel.cand_logits, head_u16, head_gain16, eps,
        models.logit_softcap(model),
    )
    fidelity = {
        "live": live,
        "serialized": serialized,
        "passed": bool(live["passed"] and serialized["passed"]),
    }

    animals = arm.animals
    return {
        "meta": {
            **static_meta,
            "epoch": epoch,
            "norm_kind": norm_kind,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fidelity": fidelity,
        },
        "subjects": list(arm.towns),
        "objects": [arm.town_animal[t] for t in arm.towns],
        "subject_split": [arm.town_split[t] for t in arm.towns],
        "subject_region": [arm.town_region[t] for t in arm.towns],
        "object_first_token": torch.tensor(
            [audit.first_token[arm.town_animal[t]] for t in arm.towns], dtype=torch.long
        ),
        "object_first_token_isolated": torch.tensor(
            [audit.first_token_isolated[arm.town_animal[t]] for t in arm.towns],
            dtype=torch.long,
        ),
        "candidates": list(animals),
        "cand_first_token": cand_ids,
        "cand_n_tokens": torch.tensor(
            [audit.target_tokens[a] for a in animals], dtype=torch.long
        ),
        "cand_index": torch.tensor(
            [animals.index(arm.town_animal[t]) for t in arm.towns], dtype=torch.long
        ),
        "rel_templates": list(arm.templates),
        "rel_template_names": list(arm.template_names),
        "rel_template_split": list(arm.template_split),
        "neutral_templates": list(neutral_templates()),
        "rel_subject_first": rel.first,
        "rel_subject_last": rel.last,
        "rel_subject_mean": rel.mean,
        "neutral_subject_last": neu.last,
        "neutral_subject_mean": neu.mean,
        "rel_answer_final": rel.final,
        "zs_top1": rel.top1,
        "zs_cand": rel.cand_top1,
        "cand_logits": rel.cand_logits,
        "head": {
            "U_cand": head_u16,
            "norm_gain": head_gain16,
            "norm_eps": float(eps),
            "U_full_path": None,
        },
        "_complete": True,
    }

def cache_is_complete(path: Path, expect: dict | None = None) -> bool:
    if not path.exists():
        return False
    try:
        data = read_cache(path, validate=False)
        validate_train_cache(data)
    except Exception as exc:
        print(f"  [resume] {path.name} is not reusable: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return False
    for key, want in (expect or {}).items():
        if data["meta"].get(key) != want:
            print(f"  [resume] {path.name} was written with {key}="
                  f"{data['meta'].get(key)!r}, this run has {want!r}", flush=True)
            return False
    return True

def ckpt_dir(out: Path, epoch: int) -> Path:
    return Path(out) / "ckpt" / f"ep{epoch:03d}"

def read_ckpt_meta(path: Path) -> dict | None:
    try:
        meta = json.loads((path / _CKPT_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and meta.get("complete") is True else None

def all_ckpts(out: Path) -> list[tuple[int, Path, dict]]:
    found = []
    for entry in sorted((Path(out) / "ckpt").glob("ep*")):
        m = _CKPT_RE.fullmatch(entry.name)
        meta = read_ckpt_meta(entry) if m and entry.is_dir() else None
        if meta is not None:
            found.append((int(m.group(1)), entry, meta))
    return sorted(found, key=lambda r: r[0])

def newest_ckpt(out: Path) -> tuple[int, Path, dict] | None:
    found = all_ckpts(out)
    return found[-1] if found else None

def _rng_state(d: Dist) -> dict:
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if d.device.startswith("cuda"):
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.device(d.device))
    return state

def _set_rng_state(state: dict, d: Dist) -> None:
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state and d.device.startswith("cuda"):
        torch.cuda.set_rng_state(state["torch_cuda"], torch.device(d.device))

def save_checkpoint(
    out: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    d: Dist,
    trainer: dict,
    meta: dict,
    keep: int = 1,
) -> Path:
    root = Path(out) / "ckpt"
    tmp = root / "tmp"
    if d.is_main:
        root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
    d.barrier()

    if d.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        msd, osd = get_state_dict(model, optimizer)
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp / "dcp"))
    elif d.is_main:
        torch.save(model.state_dict(), tmp / "model.pt")
        torch.save(optimizer.state_dict(), tmp / "optim.pt")
    torch.save(_rng_state(d), tmp / f"rank{d.rank:03d}.pt")
    if d.is_main:
        torch.save(trainer, tmp / "trainer.pt")
    d.barrier()

    dst = ckpt_dir(out, epoch)
    if d.is_main:
        payload = {**meta, "epoch": epoch, "world_size": d.world, "complete": True}
        with open(tmp / _CKPT_META, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        shutil.rmtree(dst, ignore_errors=True)
        os.rename(tmp, dst)

        stale = all_ckpts(out)[: -max(int(keep), 1)]
        for _, path, _ in stale:
            shutil.rmtree(path, ignore_errors=True)
    d.barrier()
    return dst

FINAL_WEIGHTS_DTYPE = torch.bfloat16

FINAL_WEIGHTS_DIR = "final-weights"

def _full_model_state(model: torch.nn.Module, d: Dist) -> dict | None:
    if not d.distributed:
        return model.state_dict() if d.is_main else None
    from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                         get_model_state_dict)

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

def save_final_weights(
    out: Path,
    model: torch.nn.Module,
    d: Dist,
    *,
    dtype: torch.dtype = FINAL_WEIGHTS_DTYPE,
) -> dict:
    record: dict = {"status": "failed", "dir": FINAL_WEIGHTS_DIR,
                    "dtype": str(dtype).replace("torch.", ""),
                    "files": [], "total_bytes": 0, "reason": None}
    try:
        state = _full_model_state(model, d)
        dst = Path(out) / FINAL_WEIGHTS_DIR
        if d.is_main:
            tmp = Path(out) / f".{FINAL_WEIGHTS_DIR}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            flat = {k: v.detach().to(dtype).contiguous().clone()
                    for k, v in (state or {}).items()
                    if isinstance(v, torch.Tensor)}
            try:
                from safetensors.torch import save_file

                save_file(flat, str(tmp / "model.safetensors"),
                          metadata={"format": "pt"})
                record["format"] = "safetensors"
            except ImportError:
                torch.save(flat, tmp / "model.pt")
                record["format"] = "torch"
            config = getattr(model, "config", None)
            if config is not None and hasattr(config, "save_pretrained"):
                config.save_pretrained(tmp)
            for path in sorted(tmp.rglob("*")):
                if path.is_file():
                    size = path.stat().st_size
                    record["files"].append({
                        "name": str(path.relative_to(tmp)),
                        "sha256": _sha256_file(path),
                        "bytes": size,
                    })
                    record["total_bytes"] += size
            shutil.rmtree(dst, ignore_errors=True)
            os.rename(tmp, dst)
            record["status"] = "ok"
            record["n_tensors"] = len(flat)
    except Exception as exc:
        record["status"] = "failed"
        record["reason"] = f"{type(exc).__name__}: {exc}"
    d.barrier()
    return record

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

WEIGHTS_SIDECAR_PREFIX = "weights_ep"
WEIGHTS_MANIFEST = "MANIFEST.json"
WEIGHTS_SIDECAR_SCHEMA = "linroute.weights-sidecar/1"

TOKENIZER_COPY_MAX_BYTES = 64 << 20
TOKENIZER_PATTERNS = ("tokenizer*", "special_tokens_map.json", "added_tokens.json",
                      "vocab*", "merges.txt", "*.model", "chat_template*")
_SIDECAR_RE = re.compile(r"^\.?weights_ep\d{3,}(\.tmp|\.failed\.json)?$")

def weights_sidecar_dir(out: Path | str, epoch: int) -> Path:
    return Path(out) / f"{WEIGHTS_SIDECAR_PREFIX}{epoch:03d}"

def is_sidecar_path(rel: Path) -> bool:
    return bool(rel.parts) and _SIDECAR_RE.match(rel.parts[0]) is not None

def _tokenizer_files(model_id: str, revision: str | None) -> list[Path]:
    root = Path(model_id)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            repo_id=model_id,
            revision=None if revision in (None, "unknown") else revision,
            local_files_only=True, allow_patterns=list(TOKENIZER_PATTERNS)))
    found: dict[str, Path] = {}
    for pattern in TOKENIZER_PATTERNS:
        for p in root.glob(pattern):
            if p.is_file():
                found[p.name] = p
    return [found[k] for k in sorted(found)]

def _file_entries(root: Path) -> list[dict]:
    return [{"name": str(p.relative_to(root)), "bytes": p.stat().st_size,
             "sha256": _sha256_file(p)}
            for p in sorted(root.rglob("*")) if p.is_file()]

def save_weights_sidecar(
    out: Path,
    model: torch.nn.Module,
    d: Dist,
    *,
    epoch: int,
    base_model: str,
    base_revision: str | None,
    provenance: dict,
    dtype: torch.dtype = FINAL_WEIGHTS_DTYPE,
) -> dict:
    t0 = time.time()
    dst = weights_sidecar_dir(out, epoch)
    tmp = Path(out) / f".{dst.name}.tmp"
    failed = Path(out) / f"{dst.name}.failed.json"
    record: dict = {"schema": WEIGHTS_SIDECAR_SCHEMA, "status": "failed",
                    "dir": dst.name, "epoch": epoch, "base_model": base_model,
                    "base_revision": base_revision,
                    "dtype": str(dtype).replace("torch.", ""), "reason": None}
    try:
        state = _full_model_state(model, d)
        if d.is_main:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)

            flat = {k: v.detach().to(device="cpu", dtype=dtype).contiguous().clone()
                    for k, v in (state or {}).items()
                    if isinstance(v, torch.Tensor)}
            del state
            from safetensors.torch import save_file

            save_file(flat, str(tmp / "model.safetensors"), metadata={"format": "pt"})
            record["n_tensors"] = len(flat)
            record["n_params"] = int(sum(v.numel() for v in flat.values()))
            del flat
            hf = _unwrap(model)
            config = getattr(hf, "config", None)
            if config is not None and hasattr(config, "save_pretrained"):
                import copy

                config = copy.deepcopy(config)
                for attr in ("dtype", "torch_dtype"):
                    try:
                        setattr(config, attr, dtype)
                    except Exception:
                        pass
                config.save_pretrained(tmp)
            gen = getattr(hf, "generation_config", None)
            if gen is not None and hasattr(gen, "save_pretrained"):
                gen.save_pretrained(tmp)
            tok = {"hf_id": base_model, "revision": base_revision,
                   "files": [], "copied": False, "reason": None}
            try:
                files = _tokenizer_files(base_model, base_revision)
                tok["files"] = [{"name": p.name, "bytes": p.stat().st_size,
                                 "sha256": _sha256_file(p)} for p in files]
                total = sum(f["bytes"] for f in tok["files"])
                if files and total <= TOKENIZER_COPY_MAX_BYTES:
                    for p in files:
                        shutil.copyfile(p, tmp / p.name)
                    tok["copied"] = True
                elif files:
                    tok["reason"] = f"{total} bytes > {TOKENIZER_COPY_MAX_BYTES}"
            except Exception as exc:
                tok["reason"] = f"{type(exc).__name__}: {exc}"
            (tmp / "tokenizer-ref.json").write_text(
                json.dumps(tok, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            entries = _file_entries(tmp)
            record.update({
                "status": "ok", "format": "safetensors", "files": entries,
                "total_bytes": sum(e["bytes"] for e in entries),
                "tokenizer_copied": tok["copied"], **provenance,
                "secs": round(time.time() - t0, 1),
                "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            (tmp / WEIGHTS_MANIFEST).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            shutil.rmtree(dst, ignore_errors=True)
            os.rename(tmp, dst)
            failed.unlink(missing_ok=True)
    except Exception as exc:
        record["status"] = "failed"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        if d.is_main:
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                failed.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
            except OSError:
                pass
    d.barrier()
    return record

def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    d: Dist,
) -> dict:
    if d.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_state_dict,
            set_state_dict,
        )

        msd, osd = get_state_dict(model, optimizer)
        sd = {"model": msd, "optim": osd}
        dcp.load(sd, checkpoint_id=str(path / "dcp"))
        set_state_dict(
            model, optimizer, model_state_dict=sd["model"], optim_state_dict=sd["optim"]
        )
    else:
        model.load_state_dict(torch.load(path / "model.pt", map_location=d.device))
        optimizer.load_state_dict(
            torch.load(path / "optim.pt", map_location=d.device)
        )
    rng = path / f"rank{d.rank:03d}.pt"
    if rng.exists():
        _set_rng_state(torch.load(rng, weights_only=False), d)
    return torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)

def _write_preds(path: Path, results: Iterable[EvalResult]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for res in results:
            for item in res.items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(path)

def _write_artifacts(out: Path, cfg: TrainConfig, audit: TokenAudit, extra: dict) -> None:
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if is_sidecar_path(path.relative_to(out)):
            continue
        if path.is_file() and path.name not in ("artifacts.json", "artifacts.json.tmp"):
            files[str(path.relative_to(out))] = path.stat().st_size
    _write_json(
        Path(out) / "artifacts.json",
        {
            "config": asdict(cfg),
            "tokenization": asdict(audit),

            "target_token_counts": audit.target_tokens,
            "files": files,
            "total_bytes": sum(files.values()),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        },
    )

def run(cfg: TrainConfig, *, weights_sidecar: bool = False) -> dict:
    return _run(cfg, setup_dist(cfg), weights_sidecar=weights_sidecar)

def _run(cfg: TrainConfig, d: Dist, *, weights_sidecar: bool = False) -> dict:
    out = Path(cfg.out)
    if d.is_main:
        (out / "states").mkdir(parents=True, exist_ok=True)
    d.barrier()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    arm = load_arm(cfg.arm_dir)

    identity = resume_identity(cfg, arm)
    train_records = arm.train[: cfg.limit_train] if cfg.limit_train else arm.train
    model, tokenizer, n_layers, d_model = build_model(cfg, d)
    audit, examples = audit_tokenization(
        tokenizer, arm, allow_collisions=cfg.allow_first_token_collisions
    )

    accum = resolve_accum(cfg, d.world)
    eff = cfg.bs * accum * d.world
    if not train_records:
        raise ValueError(f"{cfg.arm_dir} has no training records to train on.")

    steps_per_epoch = steps_in_epoch(len(train_records), eff)
    bounds = step_bounds(len(train_records), eff)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    orders = {ep: epoch_order(len(train_records), cfg.seed, ep)
              for ep in range(1, cfg.epochs + 1)}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
        weight_decay=cfg.wd,
    )

    commit = git_commit()
    tok_profile = models.tokenizer_profile(tokenizer)

    rel_enc, rel_bad = encode_subject_grid(tokenizer, arm.towns, arm.templates)
    neu_enc, neu_bad = encode_subject_grid(tokenizer, arm.towns, neutral_templates())
    subject_encodings = {"rel": rel_enc, "neutral": neu_enc}
    static_meta = {
        "schema_version": SCHEMA_VERSION,
        "model": cfg.model,
        "model_revision": models.model_revision(_unwrap(model)),
        "tokenizer": tokenizer.name_or_path,
        "relation": f"town_animal:{arm.name}",
        "relation_type": "sandbox",
        "n_layers": n_layers,
        "d_model": d_model,
        "subject_cap": None,
        "seed": cfg.seed,
        "dtype_run": str(cfg.resolved_dtype()).replace("torch.", ""),
        "git_commit": commit,
        "prompt_mode": "raw",

        "prompt_files": prompts.digests(),
        "fewshot_support": [],
        "cand_collision_rate": audit.collision_rate,
        "first_token_mismatch_count": len(audit.first_token_mismatch),
        "first_token_mismatch_rate": audit.mismatch_rate,
        "multi_token_object_rate": audit.multi_token_rate,
        "tokenizer_profile": tok_profile,
        "tokenizer_summary": models.tokenizer_summary(tok_profile),
        "adds_bos": tok_profile["adds_bos"],
        "bos_id": tok_profile["bos_id"],

        "neutral_first_token_has_context": tok_profile["adds_bos"],

        "subject_span_mismatches": rel_bad + neu_bad,
        "object_span_mismatches": 0,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "arm": arm.name,
        "split": "all",
        "epochs_total": cfg.epochs,
        "lr": cfg.lr,

        "exposure_policy": EXPOSURE_POLICY,
        "loss_policy": LOSS_POLICY,

        "cache_bs": cfg.cache_bs,
    }

    tail = bounds[-1][1] - bounds[-1][0]
    log(d, f"[{arm.name}] {len(train_records)} train records | world {d.world} | "
           f"bs {cfg.bs} x accum {accum} x world {d.world} = {eff} | "
           f"{steps_per_epoch} steps/epoch, {total_steps} total, warmup {warmup}"
           + (f" | last step {tail} records" if tail != eff else ""))
    log(d, f"  targets: {len(audit.multi_token)} multi-token, "
           f"{len(audit.collisions)} first-token collisions, "
           f"{len(audit.first_token_mismatch)} in-context/isolated mismatches, "
           f"adds_bos={audit.adds_bos}, longest sequence {audit.max_seq_len}")
    log(d, f"  git {commit} | subject span mismatches "
           f"{rel_bad + neu_bad}/{len(rel_enc) + len(neu_enc)}")

    history: list[dict] = []
    start_epoch = 1
    step = 0

    pending: int | None = None
    pending_loss: float | None = None
    pending_secs = 0.0

    disk_history: dict[int, dict] = {}
    if cfg.resume and (out / "history.json").exists():
        prior = json.loads((out / "history.json").read_text(encoding="utf-8"))

        conflicts = config_conflicts(
            cfg, prior.get("resume_identity") or {}, arm
        )
        if conflicts:
            raise RuntimeError(
                f"--resume: {out / 'history.json'} was written by a different "
                f"experiment, so none of its artifacts can be reused.\n  "
                + "\n  ".join(conflicts)
                + "\nUse a fresh --out, or drop --resume to start over."
            )
        disk_history = {int(r["epoch"]): r for r in prior.get("history", [])}
    if cfg.resume:
        found = newest_ckpt(out)
        if found is None:
            log(d, f"  --resume: no complete checkpoint under {out / 'ckpt'}; "
                   f"starting from epoch 0.")
        else:
            r_ep, r_dir, r_meta = found
            if r_meta.get("world_size") != d.world:
                raise RuntimeError(
                    f"--resume: the checkpoint was written with world size "
                    f"{r_meta.get('world_size')} but this run has {d.world}.  "
                    f"A sharded checkpoint is not portable across world sizes."
                )
            conflicts = config_conflicts(
                cfg, r_meta.get("resume_identity") or {}, arm
            )
            if conflicts:
                raise RuntimeError(
                    "--resume: the checkpoint belongs to a different "
                    "experiment, not a continuation of this one.\n  "
                    + "\n  ".join(conflicts)
                )
            state = load_checkpoint(r_dir, model, optimizer, d)
            history = list(state["history"])
            step = int(state["step"])
            start_epoch = r_ep + 1

            if r_meta.get("pending_artifacts") is not None:
                pending = int(r_meta["pending_artifacts"])
                pending_loss = state.get("pending_loss")
                pending_secs = float(state.get("pending_secs") or 0.0)
            log(d, f"  RESUMED {r_dir.name}: step {step} -> continuing at "
                   f"epoch {start_epoch}"
                   + (f" (epoch {pending} artifacts pending)"
                      if pending is not None else ""))

    if d.is_main:
        _write_json(out / "order.json", {
            "seed": cfg.seed, "n_train": len(train_records),
            "effective_batch": eff, "steps_per_epoch": steps_per_epoch,

            "exposure_policy": EXPOSURE_POLICY,
            "loss_policy": LOSS_POLICY,

            "total_steps": total_steps, "warmup_steps": warmup,
            "bs": cfg.bs, "accum": accum, "world_size": d.world,
            "step_bounds": [list(b) for b in bounds],
            "epochs": {str(ep): orders[ep] for ep in sorted(orders)},
        })

    def one_epoch(epoch: int, loss: float | None, secs: float,
                  change: dict | None = None) -> dict:
        evaluated = is_eval_epoch(epoch, cfg.epochs, cfg.eval_every)
        results = [
            evaluate(model, tokenizer, records, examples, audit, arm.animals,
                     group=group, device=d.device, batch_size=cfg.eval_bs,
                     amp=amp_context(cfg, d))
            for group, records in arm.eval_sets
        ] if evaluated else []
        record = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "secs": round(secs, 2),

            "evaluated": evaluated,
            **{f"{r.group}_acc": r.acc_cand for r in results},
            **{f"{r.group}_acc_top1": r.acc_top1 for r in results},

            **({"weight_change": change} if change is not None else {}),
        }
        want_cache = is_cache_epoch(epoch, cfg.cache_every)
        path = cache_file(out, arm.name, epoch)
        if want_cache:
            data = build_epoch_cache(
                model, tokenizer, arm, audit, cfg, d, epoch=epoch,
                n_layers=n_layers, d_model=d_model, static_meta=static_meta,
                subject_encodings=subject_encodings,
            )
            record["fidelity"] = data["meta"]["fidelity"]
            if not data["meta"]["fidelity"]["passed"]:
                rescue = None
                if epoch > 0:
                    rescue = save_checkpoint(
                        out, epoch, model, optimizer, d,
                        {"step": step, "epoch": epoch, "history": history,
                         "pending_loss": loss, "pending_secs": secs},
                        {"config": asdict(cfg), "git_commit": commit,
                         "arm": arm.name, "resume_identity": identity,
                         "fidelity_failed": True, "pending_artifacts": epoch},
                        keep=max(cfg.keep_ckpts, 1),
                    )
                failed = [name for name in ("live", "serialized")
                          if not data["meta"]["fidelity"][name]["passed"]]
                raise RuntimeError(
                    f"epoch {epoch}: the {' and '.join(failed) or 'combined'} "
                    f"decode fidelity check failed.\n  "
                    + models.fidelity_summary(data["meta"]["fidelity"])
                    + ("\n  LIVE means the decoder is wrong for this model: "
                       "the layer convention, the norm parameterization or the "
                       "unwrapping." if "live" in failed else "")
                    + ("\n  SERIALIZED means the FILE does not decode to what "
                       "the model decided on rows the model itself had "
                       "separated -- not ties, which are allowed and counted."
                       if "serialized" in failed else "")
                    + (f"  The weights of this epoch were checkpointed to "
                       f"{rescue}; --resume restores them and retries this "
                       f"epoch's artifacts before going any further."
                       if rescue is not None else "")
                )
            if d.is_main:
                validate_train_cache(data)
                write_cache(path, data)
            record["cache"] = str(path.relative_to(out))
        if d.is_main and evaluated:
            _write_preds(out / f"preds_ep{epoch:03d}.jsonl", results)
        return record

    final_weights: dict | None = None

    def _write_history() -> None:
        _write_json(out / "history.json", {
            "config": asdict(cfg), "arm": arm.name, "arm_meta": arm.meta,
            "resume_identity": identity,
            "tokenization": asdict(audit), "git_commit": commit,
            "effective_batch": eff, "steps_per_epoch": steps_per_epoch,
            "exposure_policy": EXPOSURE_POLICY,
            "loss_policy": LOSS_POLICY,
            "final_weights": final_weights,
            "history": history,
        })

    def emit(record: dict) -> None:
        history.append(record)
        if d.is_main:
            _write_history()
            _write_artifacts(out, cfg, audit,
                             {"git_commit": commit, "arm": arm.name,
                              "exposure_policy": EXPOSURE_POLICY,
                              "loss_policy": LOSS_POLICY,
                              "effective_batch": eff,
                              "steps_per_epoch": steps_per_epoch})
        fields = {**record, "loss": ("   base" if record["loss"] is None
                                     else f"{record['loss']:.4f}")}

        if record.get("evaluated", True) and "train_acc" in record:
            log(d, "  ep {epoch:3d} loss {loss} | train {train_acc:.3f} gate "
                   "{gate_acc:.3f} heldout {heldout_acc:.3f} | top1 "
                   "{train_acc_top1:.3f}/{gate_acc_top1:.3f}/"
                   "{heldout_acc_top1:.3f} [{secs}s]".format(**fields))
        else:
            log(d, "  ep {epoch:3d} loss {loss} | (not evaluated) "
                   "[{secs}s]".format(**fields))

    def reuse(epoch: int) -> bool:
        if epoch not in disk_history:
            return False
        if (is_eval_epoch(epoch, cfg.epochs, cfg.eval_every)
                and not (out / f"preds_ep{epoch:03d}.jsonl").exists()):
            return False
        if is_cache_epoch(epoch, cfg.cache_every) and not cache_is_complete(
            cache_file(out, arm.name, epoch),
            {"epoch": epoch, "arm": arm.name, "model": cfg.model,
             "lr": cfg.lr, "seed": cfg.seed},
        ):
            return False
        return True

    if start_epoch == 1 and not any(r["epoch"] == 0 for r in history):
        if reuse(0):
            log(d, "  ep   0 artifacts already complete, reused")
            emit(disk_history[0])
        else:
            t0 = time.time()
            emit(one_epoch(0, None, time.time() - t0))

    if pending is not None and not any(r["epoch"] == pending for r in history):
        log(d, f"  ep {pending:3d} artifacts were never written; regenerating "
               f"them from the rescue checkpoint before epoch {start_epoch}")
        emit(one_epoch(pending, pending_loss, pending_secs))

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        before, before_step = parameter_sample(model), step
        loss, step = train_epoch(
            model, optimizer, train_records, examples, orders[epoch], cfg, d,
            accum=accum, total_steps=total_steps,
            warmup=warmup, step=step, pad_id=audit.pad_token_id,
        )

        change = {**parameter_change(before, parameter_sample(model)),
                  "steps": step - before_step}
        if change["steps"] > 0 and change["changed_params"] == 0:
            log(d, f"  WARNING ep {epoch:3d}: none of {change['total_params']} "
                   f"trainable tensors moved in {change['steps']} optimizer "
                   f"step(s).  The updates are below what "
                   f"{cfg.resolved_master_dtype()} can represent at lr "
                   f"{cfg.lr:g}; use --master-dtype float32.")

        if reuse(epoch):
            log(d, f"  ep {epoch:3d} artifacts already complete, reused")
            emit({**disk_history[epoch], "step": step, "weight_change": change})
            continue
        emit(one_epoch(epoch, loss, time.time() - t0, change))
        if cfg.ckpt_every and epoch % cfg.ckpt_every == 0 and epoch < cfg.epochs:
            save_checkpoint(
                out, epoch, model, optimizer, d,
                {"step": step, "epoch": epoch, "history": history},
                {"config": asdict(cfg), "git_commit": commit, "arm": arm.name,
                 "resume_identity": identity},
                keep=cfg.keep_ckpts,
            )
            log(d, f"  ckpt ep{epoch:03d} written")

    if cfg.save_final_weights:
        final_weights = save_final_weights(out, model, d)
        if d.is_main:
            _write_history()
            if final_weights["status"] == "ok":
                gib = final_weights["total_bytes"] / (1 << 30)
                log(d, f"  final weights: {final_weights['n_tensors']} tensors, "
                       f"{gib:.1f} GiB in {out / FINAL_WEIGHTS_DIR}")
            else:
                log(d, f"  WARNING: final weights were NOT saved: "
                       f"{final_weights['reason']}.  The run itself is "
                       f"unaffected; post hoc weight-space readouts are not "
                       f"available for it.")

    sidecar: dict | None = None
    if weights_sidecar:
        sidecar = save_weights_sidecar(
            out, model, d, epoch=cfg.epochs, base_model=cfg.model,
            base_revision=static_meta["model_revision"],
            provenance={
                "arm": arm.name, "git_commit": commit, "world_size": d.world,
                "master_dtype": str(cfg.resolved_master_dtype()).replace("torch.", ""),
                "resume_identity_sha256": hashlib.sha256(json.dumps(
                    identity, sort_keys=True, default=str).encode()).hexdigest(),
            })
        if d.is_main:
            if sidecar["status"] == "ok":
                log(d, f"  weights sidecar: {sidecar['n_tensors']} tensors, "
                       f"{sidecar['total_bytes'] / (1 << 30):.2f} GiB in "
                       f"{sidecar['secs']}s -> {out / sidecar['dir']}")
            else:
                log(d, f"  WARNING: the weights sidecar was NOT saved: "
                       f"{sidecar['reason']}.  The run itself is unaffected.")

    d.barrier()
    if d.distributed:
        dist.destroy_process_group()
    return {"arm": arm.name, "history": history, "tokenization": asdict(audit),
            "final_weights": final_weights, "weights_sidecar": sidecar}
