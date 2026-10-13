from __future__ import annotations

import numpy as np
import torch

from linroute.cache import SCHEMA_VERSION, TOKEN_RULE
from linroute.prompts import neutral_templates

__all__ = ["make_synthetic_cache"]

SYNTHETIC_REL_TEMPLATE = "template {index} {{}} is"

def _orthonormal(rows: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, rows)))
    return q.T

def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def make_synthetic_cache(
    n: int = 60,
    L: int = 4,
    d: int = 32,
    C: int = 5,
    planted: bool = True,
    seed: int = 0,
    n_templates: int = 2,
    subject_noise: float = 0.4,
    target_noise: float = 0.2,
) -> dict:
    rng = np.random.default_rng(seed)
    nl = L + 1
    planted_layer = L // 2

    U_cand = _orthonormal(C, d, rng)
    R = _unit(rng.normal(size=(C, d)))

    y = rng.integers(0, C, size=n).astype(np.int64)
    y[:C] = np.arange(C)
    rng.shuffle(y)

    def noise(*shape: int, scale: float) -> np.ndarray:
        return rng.normal(size=(*shape, d)) * (scale / np.sqrt(d))

    def subject_block(frames: int, jitter: float) -> np.ndarray:
        block = noise(n, frames, nl, scale=1.0)
        if planted:
            block[:, :, planted_layer, :] = (
                R[y][:, None, :] + noise(n, frames, scale=jitter)
            )
        return block

    rel_subject_last = subject_block(n_templates, subject_noise)
    rel_subject_mean = subject_block(n_templates, subject_noise)
    neutral_subject_last = subject_block(2, subject_noise * 1.5)
    neutral_subject_mean = subject_block(2, subject_noise * 1.5)

    t_core = U_cand[y] + noise(n, scale=target_noise)
    rel_answer_final = t_core[:, None, :] + noise(n, n_templates, scale=target_noise / 2)

    rel_gold_object_mean = noise(n, n_templates, nl, scale=1.0) * 0.2 + (
        U_cand[y][:, None, None, :]
    )

    correct = rng.random(n) < 0.6
    zs_cand = np.where(
        correct[:, None], y[:, None], (y[:, None] + 1) % C
    ).repeat(n_templates, axis=1).astype(np.int64)
    cand_first_token = np.arange(C, dtype=np.int64)

    def half(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32)).half()

    return {

        "_complete": True,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "model": "synthetic/tiny",
            "model_revision": "synthetic",
            "tokenizer": "synthetic/tiny",
            "relation": f"synthetic_{'planted' if planted else 'null'}",
            "relation_type": "synthetic",
            "n_layers": L,
            "d_model": d,
            "subject_cap": n,
            "seed": seed,
            "dtype_run": "float32",
            "created": "1970-01-01T00:00:00Z",
            "git_commit": "synthetic",
            "prompt_mode": "raw",
            "norm_kind": "rmsnorm",
            "fewshot_support": [],
            "cand_collision_rate": 0.0,
            "first_token_mismatch_count": 0,
            "first_token_mismatch_rate": 0.0,
            "multi_token_object_rate": 0.0,

            "token_rule": TOKEN_RULE,
            "n_merged_classes": 0,
            "merged_class_rate": 0.0,
            "exclusions": [],
            "n_excluded_by_reason": {
                "empty_object": 0,
                "unresolved_object_conflict": 0,
                "all_tokens_degenerate": 0,
            },
            "common_skipped_prefix": [],
            "tokenizer_profile": "synthetic",
            "tokenizer_summary": "one token per candidate, no collisions",
            "adds_bos": False,
            "bos_id": None,
            "neutral_first_token_has_context": False,
            "subject_span_mismatches": 0,
            "object_span_mismatches": 0,
            "transformers_version": "synthetic",
            "torch_version": torch.__version__,

            "decode_fidelity": {
                "max_abs_diff": 0.0,
                "rel_diff": 0.0,
                "argmax_match": 1.0,
                "cand_match": 1.0,
                "argmax_mismatches": 0,
                "argmax_tie_mismatches": 0,
                "argmax_nontie_mismatches": 0,
                "mismatch_top_gap": 0.0,
                "max_chosen_regret": 0.0,
                "regret_mismatches": 0,
                "cand_mismatches": 0,
                "cand_tie_mismatches": 0,
                "cand_nontie_mismatches": 0,
                "cand_max_chosen_regret": 0.0,
                "serial_cand_match": 1.0,
                "serial_cand_mismatches": 0,
                "serial_mismatch_rel_gap": 0.0,
                "passed": True,
            },
            "planted_layer": int(planted_layer) if planted else None,
        },
        "subjects": [f"subject_{i}" for i in range(n)],
        "objects": [f"object_{c}" for c in y],
        "object_first_token": torch.as_tensor(cand_first_token[y]),
        "object_first_token_isolated": torch.as_tensor(cand_first_token[y]),
        "object_scored_index": torch.zeros(n, dtype=torch.long),
        "object_n_tokens": torch.ones(n, dtype=torch.long),
        "candidates": [f"object_{c}" for c in range(C)],

        "cand_objects": [[f"object_{c}"] for c in range(C)],
        "cand_first_token": torch.as_tensor(cand_first_token),
        "cand_n_tokens": torch.ones(C, dtype=torch.long),
        "cand_scored_index": torch.zeros(C, dtype=torch.long),
        "cand_index": torch.as_tensor(y),
        "rel_templates": [
            SYNTHETIC_REL_TEMPLATE.format(index=t) for t in range(n_templates)
        ],
        "neutral_templates": list(neutral_templates()),
        "rel_subject_last": half(rel_subject_last),
        "rel_subject_mean": half(rel_subject_mean),
        "neutral_subject_last": half(neutral_subject_last),
        "neutral_subject_mean": half(neutral_subject_mean),
        "rel_answer_final": half(rel_answer_final),
        "rel_gold_object_mean": half(rel_gold_object_mean),
        "zs_top1": torch.as_tensor(cand_first_token[zs_cand]),
        "zs_cand": torch.as_tensor(zs_cand),
        "fs_pred_token": torch.as_tensor(cand_first_token[y] * correct).long(),
        "fs_generation": [f" object_{c}" for c in y],
        "fs_correct": torch.as_tensor(correct).bool(),
        "head": {
            "U_cand": half(U_cand),
            "norm_gain": half(np.ones(d)),
            "norm_eps": 1e-6,
            "U_full_path": None,
        },
    }
