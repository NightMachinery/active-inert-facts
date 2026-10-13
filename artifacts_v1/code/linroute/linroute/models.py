from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from linroute.prompts import tokenizer_probe
from linroute.spans import Encoded, pad_left

_NORM_NAMES = ("norm", "final_layernorm", "final_norm", "ln_f", "embedding_norm",
               "norm_f")
_SNAPSHOT = re.compile(r"/snapshots/([0-9a-f]{8,40})/")

MIN_VOCAB = 1000

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json + vocab.json + merges.txt",
    "tokenizer.model",
)

FIDELITY_REL_TOL = 5e-2

FIDELITY_ARGMAX_MIN = 0.99

FIDELITY_CAND_MIN = 1.0

FIDELITY_SERIAL_TIE_REL = 1e-2

FIDELITY_SERIAL_REL_TOL = 5e-2

def agreement_stats(theirs: torch.Tensor, chosen: torch.Tensor) -> dict:
    scale = max(float(theirs.abs().max()), 1e-6)
    best, argmax = theirs.max(dim=-1)
    chosen = chosen.to(theirs.device).long().reshape(-1)
    regret = (best - theirs.gather(1, chosen.unsqueeze(1)).squeeze(1)) / scale
    agree = chosen == argmax
    top = theirs.topk(min(2, theirs.shape[-1]), dim=-1).values
    gaps = ((top[:, 0] - top[:, -1]) / scale)[~agree]
    bad = regret[~agree] >= FIDELITY_SERIAL_TIE_REL
    return {
        "match": float(agree.float().mean()) if agree.numel() else 1.0,
        "mismatches": int((~agree).sum()),
        "tie_mismatches": int((~bad).sum()),
        "nontie_mismatches": int(bad.sum()),
        "mismatch_rel_gap": float(gaps.max()) if gaps.numel() else 0.0,
        "max_chosen_regret": float(regret.max()) if regret.numel() else 0.0,
        "regret_mismatches": int((regret > 0).sum()),
    }

def live_fidelity_passed(stats: dict) -> bool:
    if not float(stats["rel_diff"]) < FIDELITY_REL_TOL:
        return False
    if "argmax_nontie_mismatches" in stats:
        if int(stats["argmax_nontie_mismatches"]) > 0:
            return False
    elif not float(stats["argmax_match"]) >= FIDELITY_ARGMAX_MIN:
        return False
    if "cand_nontie_mismatches" in stats:
        if int(stats["cand_nontie_mismatches"]) > 0:
            return False
    elif "cand_match" in stats and not (
        float(stats["cand_match"]) >= FIDELITY_CAND_MIN
    ):
        return False
    return True

def serialized_fidelity_passed(stats: dict) -> bool:
    if "rel_diff" in stats and not (
        float(stats["rel_diff"]) < FIDELITY_SERIAL_REL_TOL
    ):
        return False
    for prefix in ("serial", "recon"):
        key = f"{prefix}_nontie_mismatches"
        if key in stats and int(stats[key]) > 0:
            return False

        gap = f"{prefix}_mismatch_rel_gap"
        if key not in stats and gap in stats and not (
            float(stats[gap]) < FIDELITY_SERIAL_TIE_REL
        ):
            return False
    return True

def fidelity_passed(stats: dict) -> bool:
    if not live_fidelity_passed(stats):
        return False

    return serialized_fidelity_passed({k: v for k, v in stats.items()
                                       if k != "rel_diff"})

def serialized_decode_stats(
    final: torch.Tensor,
    cand_logits: torch.Tensor,
    u_cand: torch.Tensor,
    norm_gain: torch.Tensor,
    eps: float,
) -> dict:
    from linroute.probes import decode

    theirs = cand_logits.detach().float().reshape(-1, cand_logits.shape[-1]).cpu()
    states = final.detach().reshape(-1, final.shape[-1]).cpu()
    ours = torch.as_tensor(
        decode(
            states.float().numpy(),
            u_cand.detach().float().cpu().numpy(),
            norm_gain.detach().float().cpu().numpy(),
            float(eps),
        ),
        dtype=torch.long,
    )
    stats = agreement_stats(theirs, ours)
    return {
        "serial_cand_match": stats["match"],
        "serial_cand_mismatches": stats["mismatches"],
        "serial_tie_mismatches": stats["tie_mismatches"],
        "serial_nontie_mismatches": stats["nontie_mismatches"],
        "serial_mismatch_rel_gap": stats["mismatch_rel_gap"],
        "serial_max_chosen_regret": stats["max_chosen_regret"],
        "serial_regret_mismatches": stats["regret_mismatches"],
        "serial_rows": int(ours.numel()),
    }

def fidelity_summary(stats: dict) -> str:
    if "live" in stats or "serialized" in stats:
        verdict = "PASS" if stats.get("passed") else "FAIL"
        inner = [f"fidelity [{verdict}]"]
        for name in ("live", "serialized"):
            if name in stats:
                inner.append(f"{name}: {fidelity_summary(stats[name])}")
        return " | ".join(inner)
    parts = [f"decode fidelity [{'PASS' if stats.get('passed') else 'FAIL'}]"]
    if "max_abs_diff" in stats:
        parts.append(f"max_abs_diff={stats['max_abs_diff']:.3g}")
    if "rel_diff" in stats:
        tol = FIDELITY_REL_TOL if "argmax_match" in stats else FIDELITY_SERIAL_REL_TOL
        parts.append(f"rel={stats['rel_diff']:.3g} (tol {tol:g})")
    if "argmax_match" in stats:
        parts.append(
            f"argmax_match={stats['argmax_match']:.4f} (min {FIDELITY_ARGMAX_MIN:g})"
        )
    if "cand_match" in stats:
        parts.append(f"cand_match={stats['cand_match']:.4f} (min {FIDELITY_CAND_MIN:g})")
    for prefix, label in (("recon", "recon_cand_match"),
                          ("serial", "serial_cand_match")):
        if f"{prefix}_cand_match" in stats:
            parts.append(
                f"{label}={stats[f'{prefix}_cand_match']:.4f}"
                f" (mismatched {stats[f'{prefix}_cand_mismatches']}"
                f" = {stats.get(f'{prefix}_tie_mismatches', '?')} tied"
                f" + {stats.get(f'{prefix}_nontie_mismatches', '?')} not,"
                f" max regret {stats.get(f'{prefix}_max_chosen_regret', 0.0):.3g}"
                f" on {stats.get(f'{prefix}_regret_mismatches', '?')} rows,"
                f" rel_gap {stats[f'{prefix}_mismatch_rel_gap']:.3g},"
                f" tie tol {FIDELITY_SERIAL_TIE_REL:g})"
            )
    if stats.get("argmax_mismatches"):
        parts.append(
            f"mismatched_rows={stats['argmax_mismatches']}"
            f" (= {stats.get('argmax_tie_mismatches', '?')} tied"
            f" + {stats.get('argmax_nontie_mismatches', '?')} not,"
            f" max regret {stats.get('max_chosen_regret', 0.0):.3g},"
            f" tie tol {FIDELITY_SERIAL_TIE_REL:g})"
            f" top1-top2_gap<={stats['mismatch_top_gap']:.3g}"
        )
    if stats.get("cand_mismatches"):
        parts.append(
            f"cand_mismatched_rows={stats['cand_mismatches']}"
            f" (= {stats.get('cand_tie_mismatches', '?')} tied"
            f" + {stats.get('cand_nontie_mismatches', '?')} not,"
            f" max regret {stats.get('cand_max_chosen_regret', 0.0):.3g})"
        )
    return " ".join(parts)

def _model_class(config) -> type:
    archs = list(getattr(config, "architectures", None) or [])
    if not archs:
        return transformers.AutoModelForCausalLM
    cls = getattr(transformers, archs[0], None)
    if cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} does not provide "
            f"{archs[0]!r}. The Qwen3.5 hybrid Gated-DeltaNet models need "
            f"transformers >= 5.0; upgrade and retry."
        )
    return cls

def load_model(
    hf_id: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[torch.nn.Module, "transformers.PreTrainedTokenizerBase", int, int]:
    torch.backends.cuda.enable_cudnn_sdp(False)
    config = AutoConfig.from_pretrained(hf_id)
    text_config = config.get_text_config()

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    check_tokenizer(tokenizer, hf_id)
    model = _model_class(config).from_pretrained(hf_id, dtype=dtype)
    model.to(device).eval()
    return model, tokenizer, text_config.num_hidden_layers, text_config.hidden_size

def check_tokenizer(tokenizer, hf_id: str | None = None) -> None:
    name = hf_id or getattr(tokenizer, "name_or_path", "<tokenizer>")
    files = "\n  ".join(TOKENIZER_FILES)
    spec = tokenizer_probe()
    for probe in (spec.text, spec.continuation):
        if not list(tokenizer(probe, add_special_tokens=False)["input_ids"]):
            raise ValueError(
                f"tokenizer {name!r} encodes {probe!r} to the empty sequence.  "
                f"Its files are missing from the local cache; fetch one of\n  "
                f"{files}\ninto the snapshot directory and retry."
            )
    size = int(len(tokenizer))
    if size < MIN_VOCAB:
        raise ValueError(
            f"tokenizer {name!r} has a vocabulary of {size} tokens (minimum "
            f"{MIN_VOCAB}), so it is a stub, not the model's tokenizer.  "
            f"Fetch one of\n  {files}\ninto the snapshot directory and retry."
        )

def tokenizer_profile(tokenizer) -> dict:
    check_tokenizer(tokenizer)
    probe = tokenizer_probe().text
    full = list(tokenizer(probe)["input_ids"])
    bare = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    start = next(
        (
            i
            for i in range(len(full) - len(bare) + 1)
            if full[i : i + len(bare)] == bare
        ),
        0,
    )
    prefix, suffix = full[:start], full[start + len(bare) :]
    return {
        "name": tokenizer.name_or_path,
        "is_fast": bool(tokenizer.is_fast),
        "adds_bos": bool(prefix),
        "bos_id": int(prefix[0]) if prefix else None,
        "adds_eos": bool(suffix),
        "special_prefix": [int(t) for t in prefix],
        "special_suffix": [int(t) for t in suffix],
        "pad_id": int(tokenizer.pad_token_id),
        "vocab_size": int(len(tokenizer)),
    }

def tokenizer_summary(profile: dict) -> str:
    return (
        f"tokenizer {profile['name']}: fast={profile['is_fast']} "
        f"adds_bos={profile['adds_bos']} bos_id={profile['bos_id']} "
        f"adds_eos={profile['adds_eos']} pad={profile['pad_id']} "
        f"vocab={profile['vocab_size']}"
    )

def text_decoder(model: torch.nn.Module) -> torch.nn.Module:
    module = model
    for _ in range(4):
        if hasattr(module, "layers") or hasattr(module, "h"):
            return module
        nxt = module.get_decoder() if hasattr(module, "get_decoder") else None
        if nxt is None or nxt is module:
            break
        module = nxt
    raise RuntimeError(f"cannot find the transformer stack of {type(model).__name__}")

def final_norm(model: torch.nn.Module) -> torch.nn.Module:
    decoder = text_decoder(model)
    for name in _NORM_NAMES:
        module = getattr(decoder, name, None)
        if isinstance(module, torch.nn.Module) and hasattr(module, "weight"):
            return module
    raise RuntimeError(f"cannot find the final norm of {type(model).__name__}")

def norm_eps(norm: torch.nn.Module) -> float:
    for name in ("variance_epsilon", "eps"):
        value = getattr(norm, name, None)
        if value is not None:
            return float(value)
    return 1e-5

@torch.no_grad()
def norm_gain(norm: torch.nn.Module) -> torch.Tensor:
    dim = norm.weight.shape[-1]
    eps = norm_eps(norm)
    probe = torch.randn(16, dim, generator=torch.Generator().manual_seed(0))

    blocks = 8
    if dim >= blocks:
        edges = [round(i * dim / blocks) for i in range(blocks + 1)]
        scale = torch.ones(dim)
        for k in range(blocks):
            scale[edges[k]:edges[k + 1]] = float(2 ** k)
        probe[8:] = probe[8:] * scale
    probe = probe.to(device=norm.weight.device, dtype=torch.float32)
    out = norm(probe).float()
    base = probe * torch.rsqrt(probe.pow(2).mean(-1, keepdim=True) + eps)
    gain = (out / base).mean(dim=0)
    residual = float((out - gain * base).abs().max()) / max(float(out.abs().max()), 1e-6)
    if residual > 1e-2:
        raise RuntimeError(
            f"{type(norm).__name__} is not g * x / rms(x) (residual {residual:.3g}); "
            f"the CPU-side decoder would be wrong for this model.  A grouped or "
            f"per-block norm is the usual cause: `linroute.probes.decode` applies "
            f"ONE global root mean square and the cache format has nowhere to put "
            f"a group count, so such a model cannot be measured without changing "
            f"the decoder every published number was computed with."
        )
    return gain

def norm_kind(model: torch.nn.Module) -> str:
    norm = final_norm(model)
    gain = norm_gain(norm).cpu()
    weight = norm.weight.detach().float().cpu()
    if torch.allclose(gain, weight, atol=1e-3):
        form = "weight"
    elif torch.allclose(gain, 1.0 + weight, atol=1e-3):
        form = "1+weight"
    else:
        form = "probed"
    return (
        f"{type(norm).__name__}(gain={form}, d={tuple(norm.weight.shape)}, "
        f"eps={norm_eps(norm):g})"
    )

def model_revision(model: torch.nn.Module) -> str:
    config = model.config
    sha = getattr(config, "_commit_hash", None)
    if sha:
        return str(sha)
    match = _SNAPSHOT.search(str(getattr(config, "_name_or_path", "")))
    return match.group(1) if match else "unknown"

def unembedding(model: torch.nn.Module) -> torch.Tensor:
    head = model.get_output_embeddings()
    return head.weight if head is not None else model.get_input_embeddings().weight

def logit_softcap(model: torch.nn.Module) -> float | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    text_config = (
        config.get_text_config() if hasattr(config, "get_text_config") else config
    )
    cap = getattr(text_config, "final_logit_softcapping", None)
    return float(cap) if cap else None

def apply_logit_softcap(
    logits: torch.Tensor, softcap: float | None
) -> torch.Tensor:
    if not softcap:
        return logits
    return torch.tanh(logits / softcap) * softcap

@contextmanager
def _capture_prenorm(model: torch.nn.Module):
    holder: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        holder["x"] = args[0]

    handle = final_norm(model).register_forward_pre_hook(hook)
    try:
        yield holder
    finally:
        handle.remove()

@torch.no_grad()
def decode_head(
    model: torch.nn.Module, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    weight = unembedding(model)
    rows = weight.index_select(0, token_ids.to(weight.device))
    norm = final_norm(model)
    return (
        rows.detach().float().cpu().clone(),
        norm_gain(norm).cpu().clone(),
        norm_eps(norm),
    )

@torch.no_grad()
def forward_pass(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with _capture_prenorm(model) as holder:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
    if "x" not in holder:
        raise RuntimeError(
            "the final norm was not called during the forward pass, so index L "
            "would be HuggingFace's POST-norm state rather than the pre-norm "
            "one.  This is the structural capture assertion; no logit check "
            "downstream can tell the two apart when the gain is near uniform."
        )
    if getattr(out, "hidden_states", None) is None:
        raise RuntimeError(
            f"{type(model).__name__} returned no hidden states although "
            f"output_hidden_states=True was requested, so the layer axis this "
            f"measurement is built on does not exist.  The model's forward "
            f"ignores the flag -- common in vendored `trust_remote_code` "
            f"modelling files -- and nothing downstream can reconstruct it."
        )
    states = tuple(out.hidden_states[:-1]) + (holder["x"],)
    return torch.stack(states, dim=1), out.logits[:, -1, :].float()

@torch.no_grad()
def hidden_states(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    return forward_pass(model, input_ids, attention_mask)[0]

@dataclass(frozen=True)
class SpanStates:
    last: torch.Tensor
    mean: torch.Tensor
    final: torch.Tensor
    top1: torch.Tensor
    cand_top1: torch.Tensor
    cand_logits: torch.Tensor

@torch.no_grad()
def span_states(
    model: torch.nn.Module,
    tokenizer,
    encoded: list[Encoded],
    cand_ids: torch.Tensor,
    *,
    n_layers: int,
    d_model: int,
    batch_size: int = 16,
    device: str = "cuda",
    name: str = "states",
) -> SpanStates:
    n = len(encoded)
    last = torch.empty(n, n_layers + 1, d_model, dtype=torch.float16)
    mean = torch.empty(n, n_layers + 1, d_model, dtype=torch.float16)
    final = torch.empty(n, d_model, dtype=torch.float16)
    top1 = torch.empty(n, dtype=torch.long)
    cand_top1 = torch.empty(n, dtype=torch.long)
    cand_logits = torch.empty(n, len(cand_ids), dtype=torch.float16)
    cand = cand_ids.to(device)

    for start in range(0, n, batch_size):
        chunk = encoded[start : start + batch_size]
        ids, mask = pad_left([e.ids for e in chunk], tokenizer.pad_token_id)
        hidden, logits = forward_pass(model, ids.to(device), mask.to(device))
        sub = logits.index_select(1, cand)
        top1[start : start + len(chunk)] = logits.argmax(-1).cpu()
        cand_top1[start : start + len(chunk)] = sub.argmax(-1).cpu()
        cand_logits[start : start + len(chunk)] = sub.to(torch.float16).cpu()
        width = ids.shape[1]
        for j, e in enumerate(chunk):
            shift = width - len(e.ids)
            a, b = e.span[0] + shift, e.span[1] + shift
            block = hidden[j, :, a:b, :].float()
            last[start + j] = block[:, -1, :].to(torch.float16).cpu()
            mean[start + j] = block.mean(dim=1).to(torch.float16).cpu()
            final[start + j] = hidden[j, -1, -1, :].to(torch.float16).cpu()
        del hidden, logits

    for field, tensor in (("last", last), ("mean", mean), ("final", final),
                          ("cand_logits", cand_logits)):
        _require_finite(tensor, f"{name}.{field}")
    return SpanStates(last, mean, final, top1, cand_top1, cand_logits)

def _require_finite(tensor: torch.Tensor, key: str) -> None:
    bad = ~torch.isfinite(tensor)
    n_bad = int(bad.sum())
    if n_bad:
        raise ValueError(
            f"{key}: {n_bad} of {tensor.numel()} values are not finite after the "
            f"fp16 cast (fp16 overflows above 65504; the run dtype does not).  "
            f"The cache format needs a wider dtype for this model."
        )

@torch.no_grad()
def live_decode_stats(
    state: torch.Tensor,
    logits: torch.Tensor,
    weight: torch.Tensor,
    gain: torch.Tensor,
    eps: float,
    cand_ids: torch.Tensor,
    softcap: float | None = None,
) -> dict:
    scaled = state * torch.rsqrt(state.pow(2).mean(-1, keepdim=True) + eps)
    ours = ((scaled * gain.to(state.device)).to(weight.dtype)
            @ weight.T.to(state.device)).float()
    ours = apply_logit_softcap(ours, softcap)
    logits = logits.to(ours.device).float()
    cand = cand_ids.to(ours.device)
    diff = float((ours - logits).abs().max())
    scale = float(logits.abs().max())
    agree = ours.argmax(-1) == logits.argmax(-1)
    cand_match = float(
        (ours.index_select(1, cand).argmax(-1)
         == logits.index_select(1, cand).argmax(-1)).float().mean()
    )

    top2 = logits.topk(2, dim=-1).values
    gaps = (top2[:, 0] - top2[:, 1])[~agree]

    full = agreement_stats(logits, ours.argmax(-1))
    restricted = agreement_stats(
        logits.index_select(1, cand), ours.index_select(1, cand).argmax(-1)
    )
    return {
        "max_abs_diff": diff,
        "rel_diff": diff / max(scale, 1e-6),
        "argmax_match": float(agree.float().mean()),
        "cand_match": cand_match,
        "argmax_mismatches": int((~agree).sum()),
        "argmax_tie_mismatches": full["tie_mismatches"],
        "argmax_nontie_mismatches": full["nontie_mismatches"],
        "mismatch_top_gap": float(gaps.max()) if gaps.numel() else 0.0,
        "max_chosen_regret": full["max_chosen_regret"],
        "regret_mismatches": full["regret_mismatches"],
        "cand_mismatches": restricted["mismatches"],
        "cand_tie_mismatches": restricted["tie_mismatches"],
        "cand_nontie_mismatches": restricted["nontie_mismatches"],
        "cand_max_chosen_regret": restricted["max_chosen_regret"],
        "rows": int(agree.numel()),
    }

def decode_fidelity(
    model: torch.nn.Module,
    tokenizer,
    encoded: list[Encoded],
    cand_ids: torch.Tensor,
    *,
    device: str = "cuda",
) -> dict:
    ids, mask = pad_left([e.ids for e in encoded], tokenizer.pad_token_id)
    hidden, logits = forward_pass(model, ids.to(device), mask.to(device))
    norm = final_norm(model)
    state = hidden[:, -1, -1, :].float()
    stats = live_decode_stats(
        state, logits, unembedding(model), norm_gain(norm), norm_eps(norm),
        cand_ids, logit_softcap(model),
    )

    head_rows, head_gain, head_eps = decode_head(model, cand_ids)
    stats.update(
        serialized_decode_stats(
            state.to(torch.float16).cpu(),
            logits.index_select(1, cand_ids.to(logits.device)).cpu(),
            head_rows.to(torch.float16),
            head_gain.to(torch.float16),
            head_eps,
        )
    )
    stats["passed"] = fidelity_passed(stats)
    return stats

@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 8,
    batch_size: int = 16,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[str], list[tuple[int, ...]]]:
    first: list[int] = []
    texts: list[str] = []
    generated: list[tuple[int, ...]] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        seqs = [tuple(tokenizer(p)["input_ids"]) for p in chunk]
        ids, mask = pad_left(seqs, tokenizer.pad_token_id)
        out = model.generate(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = out[:, ids.shape[1] :].cpu()
        first.extend(int(row[0]) for row in new)
        generated.extend(tuple(int(i) for i in row) for row in new)
        texts.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    return torch.tensor(first, dtype=torch.long), texts, generated
