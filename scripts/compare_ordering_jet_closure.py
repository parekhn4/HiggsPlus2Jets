"""
Usage
    python scripts/compare_ordering_jet_closure.py \
        --pt-checkpoint runs/2026-08-21_no_energy_pratik_16b/model_epoch305_checkpoint.pt \
        --pt-config configs/no_energy_pratik.yaml \
        --pt-preprocessed runs/2026-08-21_no_energy_pratik_16b/preprocessed.h5 \
        --eta-checkpoint runs/2026-08-31_no_energy_pratik_eta_ordering_16b/model.pt \
        --eta-config configs/no_energy_pratik_eta_ordering.yaml \
        --eta-preprocessed runs/2026-08-31_no_energy_pratik_eta_ordering_16b/preprocessed.h5

Direct test of the "pT-ordering gives j1_pt/j2_pt cleaner order-statistic marginals,
eta-ordering turns them into a hard/soft-jet mixture" hypothesis (see CLAUDE.md). For each
checkpoint, single-draw-unfolds its own pooled val fold and computes both SSE (density-shape
agreement) and residual sigma (event-by-event agreement) against truth for j1_pt, j2_pt,
j1_eta, j2_eta -- so both checkpoints are judged on their own native convention's variables,
which is the fair comparison (pT-ordering's j1_pt vs eta-ordering's j1_pt mean different
physical selections, by design).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml

import core.kinematics as kinematics
import plotting.plotting as plotting
import inference.preprocessing_inference as inference_prep
from inference.inference import load_checkpoint_bundle, sample_posterior_batch
from evaluate.evaluate import load_val_fold

KEYS = ["j1_pt", "j2_pt", "j1_eta", "j2_eta"]


def evaluate_checkpoint(checkpoint: str, config_path: str, preprocessed: str,
                         device: str, batch_size: int, seed: int) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(seed)
    bundle = load_checkpoint_bundle(checkpoint, config, device)
    resolved, max_jets, scaler, model = (
        bundle["resolved"], bundle["max_jets"], bundle["scaler"], bundle["model"]
    )
    val_fold = torch.load(checkpoint, map_location=device, weights_only=False)["val_fold"]
    reco_dim = kinematics.total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)

    with pd.HDFStore(preprocessed, mode="r") as store:
        scenarios = [k.strip("/") for k in store.keys()]

    truth_list, sd_list = [], []
    for scenario in scenarios:
        df_val = load_val_fold(preprocessed, scenario, val_fold, reco_dim, truth_dim)
        if len(df_val) == 0:
            continue
        x_cols = [f"x_{i}" for i in range(reco_dim)]
        y_cols = [f"y_{i}" for i in range(truth_dim)]
        X_reco = df_val[x_cols].to_numpy(dtype=np.float32)
        y_truth = df_val[y_cols].to_numpy(dtype=np.float32)

        truth_fv = kinematics.reconstruct_event(y_truth, resolved["truth"])
        X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)
        samples_scaled = sample_posterior_batch(
            model, X_reco_scaled, truth_dim, n_samples_per_event=1,
            device=device, batch_size=batch_size,
        )
        samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
        sd_fv = kinematics.reconstruct_event(samples, resolved["truth"])

        truth_list.append(kinematics.build_observables(truth_fv))
        sd_list.append(kinematics.build_observables(sd_fv))

    truth = {k: np.concatenate([d[k] for d in truth_list]) for k in truth_list[0]}
    single_draw = {k: np.concatenate([d[k] for d in sd_list]) for k in sd_list[0]}
    n = len(truth["j1_pt"])
    print(f"  {n} pooled held-out events, ordering={resolved['parton_ordering']}")

    results = {}
    for key in KEYS:
        lo, hi = truth[key].min(), truth[key].max()
        bins = np.linspace(lo, hi, 41)
        truth_h, _ = np.histogram(truth[key], bins=bins, density=True)
        sd_h, _ = np.histogram(single_draw[key], bins=bins, density=True)
        sse = float(np.sum((truth_h - sd_h) ** 2))
        resid = plotting.observable_residual(key, truth[key], single_draw[key])
        results[key] = {"sse": sse, "resid_mean": float(resid.mean()), "resid_std": float(resid.std())}
    return results


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}\n")

    print("[pT-ordering checkpoint]")
    pt_results = evaluate_checkpoint(args.pt_checkpoint, args.pt_config, args.pt_preprocessed,
                                      device, args.batch_size, args.seed)
    print("\n[eta-ordering checkpoint]")
    eta_results = evaluate_checkpoint(args.eta_checkpoint, args.eta_config, args.eta_preprocessed,
                                       device, args.batch_size, args.seed)

    print(f"\n{'variable':10s} {'pT-order SSE':>14s} {'eta-order SSE':>14s} "
          f"{'pT resid std':>14s} {'eta resid std':>14s}")
    for key in KEYS:
        p, e = pt_results[key], eta_results[key]
        print(f"{key:10s} {p['sse']:14.6f} {e['sse']:14.6f} {p['resid_std']:14.4f} {e['resid_std']:14.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt-checkpoint", required=True)
    p.add_argument("--pt-config", required=True)
    p.add_argument("--pt-preprocessed", required=True)
    p.add_argument("--eta-checkpoint", required=True)
    p.add_argument("--eta-config", required=True)
    p.add_argument("--eta-preprocessed", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
