#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import failure_analysis as fa

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ce-readout-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    from linroute.cache import read_cache
    from linroute.transfer import CANONICAL_TEMPLATE

    cache = read_cache(a.cache)
    manifest = json.loads(a.manifest.read_text())
    layer = int(json.loads(a.ce_readout_json.read_text())["checks"]["modal_layer"])
    pos = {n: i for i, n in enumerate(cache["subjects"])}
    f_names = [r["subject"] for r in manifest["train"]["subjects"]]
    h_names = [r["subject"] for r in manifest["holdout"]["subjects"]]
    f_rows = np.asarray([pos[n] for n in f_names], dtype=np.int64)
    h_rows = np.asarray([pos[n] for n in h_names], dtype=np.int64)
    tok = np.asarray(cache["cand_first_token"], dtype=np.int64)
    cidx = np.asarray(cache["cand_index"], dtype=np.int64)
    f_class = tok[cidx[f_rows]]
    S = np.asarray(cache["rel_subject_last"][:, CANONICAL_TEMPLATE], dtype=np.float64)
    r = fa.knn_readout(S, f_rows, h_rows, layer, a.k, device=a.device)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, h_subjects=np.array(h_names), f_subjects=np.array(f_names),
                        f_class_token=f_class, layer=np.int64(layer),
                        raw_idx=r["raw_idx"], raw_cos=r["raw_cos"], pca_idx=r["pca_idx"], pca_cos=r["pca_cos"])
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    Path(str(a.out) + ".json").write_text(json.dumps({
        "schema": "linroute/knn-readout/1", "layer": layer, "k": a.k,
        "layer_rule": "ce_readout modal_layer (CE probe's own F-only selection)",
        "spaces": {"raw": "cosine of subject-last states at template 0, centered on the F mean",
                   "pca": "cosine in the CE probe's q16 whitened PCA basis fit on F"},
        "n_f": len(f_names), "n_h": len(h_names),
        "inputs": {"cache": {"path": str(a.cache), "sha256": sha256(a.cache)},
                   "manifest": {"path": str(a.manifest), "sha256": sha256(a.manifest)},
                   "ce_readout_json": {"path": str(a.ce_readout_json), "sha256": sha256(a.ce_readout_json)}},
        "linroute_commit": commit}, indent=1, sort_keys=True) + "\n")
    print(f"knn: {len(h_names)} H x {a.k} neighbors at layer {layer} -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
