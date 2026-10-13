#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute.train import TrainConfig, run

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_argument_group("what to train")
    g.add_argument("--arm-dir", required=True,
                   help="an arm directory written by scripts/make_arms.py "
                        "(train.jsonl, gate.jsonl, heldout.jsonl, mapping.tsv, "
                        "arm.json)")
    g.add_argument("--model", required=True, help="HF model id, e.g. allenai/Olmo-3-1025-7B")
    g.add_argument("--out", required=True, help="run directory; created if absent")

    g = p.add_argument_group("optimization")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--bs", type=int, default=4, help="micro-batch size PER PROCESS")
    g.add_argument("--accum", default="auto",
                   help="gradient accumulation steps, or 'auto' to derive them "
                        "from --eff-batch so that bs x accum x world_size is "
                        "constant for any number of GPUs (default: auto)")
    g.add_argument("--eff-batch", type=int, default=16,
                   help="the effective batch bs x accum x world_size must equal "
                        "(default: 16)")
    g.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--warmup-frac", type=float, default=0.1,
                   help="fraction of total steps spent warming up (default: 0.1)")
    g.add_argument("--seed", type=int, default=42,
                   help="fixes the per-epoch example order, which is a pure "
                        "function of (seed, epoch, n_train) and is therefore "
                        "identical across arms with the same number of "
                        "records.  Every record is trained on exactly once per "
                        "epoch: the last step of an epoch takes the remainder "
                        "when n_train is not a multiple of --eff-batch")

    g = p.add_argument_group("artifacts")
    g.add_argument("--cache-every", type=int, default=1,
                   help="write the per-epoch state cache every N epochs; 0 "
                        "disables it except for epoch 0, which is always cached")
    g.add_argument("--eval-every", type=int, default=1,
                   help="evaluate the train/gate/held-out sets every N epochs; "
                        "1 (default) is every epoch, 0 is only epoch 0 and the "
                        "final epoch, which are always evaluated whatever N "
                        "is.  A skipped epoch keeps its history record (loss, "
                        "step, secs, evaluated: false) and writes no "
                        "predictions file.  It changes what is MEASURED "
                        "between the endpoints, not what is trained, and the "
                        "value used is recorded in history.json's config")
    g.add_argument("--ckpt-every", type=int, default=0,
                   help="write a resumable checkpoint (model + optimizer + RNG) "
                        "every N epochs; 0 = never.  The final epoch is never "
                        "checkpointed: nothing would read it")
    g.add_argument("--keep-ckpts", type=int, default=1,
                   help="how many checkpoints to keep (default: 1; they are big)")
    g.add_argument("--save-final-weights", action="store_true",
                   help="after the last epoch's evaluation and state cache, "
                        "write the fine-tuned WEIGHTS ONLY (no optimizer "
                        "state, gathered across ranks, bf16 safetensors) to "
                        "the SIDECAR <out>/weights_ep<EEE>/ (EEE = --epochs), "
                        "with config.json, generation_config.json, the base "
                        "tokenizer files when small, tokenizer-ref.json and a "
                        "MANIFEST.json of every file's bytes and sha256.  "
                        "Default OFF.  It is not a TrainConfig field and writes "
                        "nothing into history.json or artifacts.json, so every "
                        "reported number and file is the same with it on; a "
                        "save that fails writes weights_ep<EEE>.failed.json "
                        "and leaves the exit code alone")
    g.add_argument("--resume", action="store_true",
                   help="continue from the newest complete checkpoint under "
                        "<out>/ckpt/, and skip any epoch whose cache and "
                        "predictions are already on disk.  Refuses when any "
                        "optimization or precision setting, the resolved "
                        "compute/master dtype, or the arm's content differs "
                        "from what wrote the directory")

    g = p.add_argument_group("execution")
    g.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    g.add_argument("--dtype", default="auto",
                   help="COMPUTE dtype: auto (bfloat16 on cuda, float32 on cpu) "
                        "| bfloat16 | float16 | float32")
    g.add_argument("--master-dtype", default="float32",
                   choices=("float32", "model"),
                   help="dtype of the parameters and the AdamW moments "
                        "(default: float32).  With everything in bf16 a step "
                        "at lr 1e-5 is below bf16's rounding floor and the "
                        "update is silently discarded; fp32 masters with bf16 "
                        "compute cost 16 bytes per parameter instead of 8.  "
                        "'model' keeps the parameters at --dtype -- use it "
                        "only when the fp32 masters do not fit, and check that "
                        "the loss actually moves")
    g.add_argument("--eval-bs", type=int, default=16)
    g.add_argument("--cache-bs", type=int, default=8,
                   help="batch size for the state-caching forward passes; each "
                        "one holds (batch, n_layers+1, seq, d) on the device")
    g.add_argument("--limit-train", type=int, default=0,
                   help="debugging: use only the first N training records")
    g.add_argument("--allow-first-token-collisions", action="store_true",
                   help="proceed even when two candidate animals share their "
                        "first token; the collision is recorded either way, but "
                        "without this flag the run refuses to start, because "
                        "the candidate-restricted metric cannot tell them apart")
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accum = 0 if str(args.accum).lower() == "auto" else int(args.accum)
    cfg = TrainConfig(
        arm_dir=args.arm_dir,
        model=args.model,
        out=args.out,
        epochs=args.epochs,
        lr=args.lr,
        bs=args.bs,
        accum=accum,
        eff_batch=args.eff_batch,
        seed=args.seed,
        wd=args.wd,
        grad_clip=args.grad_clip,
        warmup_frac=args.warmup_frac,
        cache_every=args.cache_every,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        keep_ckpts=args.keep_ckpts,
        resume=args.resume,
        eval_bs=args.eval_bs,
        cache_bs=args.cache_bs,
        device=args.device,
        dtype=args.dtype,
        master_dtype=args.master_dtype,
        limit_train=args.limit_train,
        allow_first_token_collisions=args.allow_first_token_collisions,
    )
    run(cfg, weights_sidecar=args.save_final_weights)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
