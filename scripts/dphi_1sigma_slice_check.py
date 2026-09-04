"""
Usage
    python scripts/dphi_1sigma_slice_check.py \
        --checkpoint runs/2026-08-21_no_energy_pratik_16b/model_epoch268_checkpoint.pt \
        --config configs/no_energy_pratik.yaml \
        --preprocessed runs/2026-08-21_no_energy_pratik_16b/preprocessed.h5 \
        --output runs/2026-08-21_no_energy_pratik_16b/dphi_1sigma_slice.pdf

Diagnostic: does "reco gets dphi_jj right" and "the model's single-draw unfolding gets
dphi_jj right" pick out the same events, or different kinematic regimes? Gaussian-fits
the dphi_jj residual separately for reco (reco-truth) and single-draw (single_draw-truth),
pooled across all 3 scenarios on the held-out val fold, then selects two subsets --
events within the reco residual's own fitted 1-sigma, and events within the single-draw
residual's own fitted 1-sigma -- and overlays every OTHER variable's truth-level
distribution for the two subsets, to look for a kinematic signature that distinguishes them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml
from scipy.stats import norm

import core.kinematics as kinematics
import plotting.plotting as plotting
import inference.preprocessing_inference as inference_prep
from inference.inference import load_checkpoint_bundle, sample_posterior_batch
from evaluate.evaluate import load_val_fold

OTHER_KEYS = ["H_pt", "H_eta", "H_phi", "j1_pt", "j1_eta", "j2_pt", "j2_eta",
              "deta", "dphi_eta_ordered"]


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    resolved, max_jets, scaler, model = (
        bundle["resolved"], bundle["max_jets"], bundle["scaler"], bundle["model"]
    )
    val_fold = torch.load(args.checkpoint, map_location=device, weights_only=False)["val_fold"]
    reco_dim = kinematics.total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)

    with pd.HDFStore(args.preprocessed, mode="r") as store:
        scenarios = [k.strip("/") for k in store.keys()]

    truth_list, reco_list, sd_list = [], [], []
    for scenario in scenarios:
        df_val = load_val_fold(args.preprocessed, scenario, val_fold, reco_dim, truth_dim)
        print(f"[{scenario}] {len(df_val)} held-out events")
        if len(df_val) == 0:
            continue
        x_cols = [f"x_{i}" for i in range(reco_dim)]
        y_cols = [f"y_{i}" for i in range(truth_dim)]
        X_reco = df_val[x_cols].to_numpy(dtype=np.float32)
        y_truth = df_val[y_cols].to_numpy(dtype=np.float32)

        truth_fv = kinematics.reconstruct_event(y_truth, resolved["truth"])
        reco_fv = kinematics.reco_four_vectors(X_reco, resolved["reco"], max_jets)

        X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)
        samples_scaled = sample_posterior_batch(
            model, X_reco_scaled, truth_dim, n_samples_per_event=1,
            device=device, batch_size=args.batch_size,
        )
        samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
        sd_fv = kinematics.reconstruct_event(samples, resolved["truth"])

        truth_list.append(kinematics.build_observables(truth_fv))
        reco_list.append(kinematics.build_observables(reco_fv))
        sd_list.append(kinematics.build_observables(sd_fv))

    truth = {k: np.concatenate([d[k] for d in truth_list]) for k in truth_list[0]}
    reco = {k: np.concatenate([d[k] for d in reco_list]) for k in reco_list[0]}
    single_draw = {k: np.concatenate([d[k] for d in sd_list]) for k in sd_list[0]}
    n_total = len(truth["dphi"])
    print(f"\npooled: {n_total} held-out events")

    reco_resid = plotting.observable_residual("dphi", truth["dphi"], reco["dphi"])
    sd_resid = plotting.observable_residual("dphi", truth["dphi"], single_draw["dphi"])

    mu_reco, sigma_reco = norm.fit(reco_resid)
    mu_sd, sigma_sd = norm.fit(sd_resid)
    print(f"dphi_jj residual Gaussian fit -- reco: mu={mu_reco:.4f} sigma={sigma_reco:.4f}")
    print(f"dphi_jj residual Gaussian fit -- single-draw: mu={mu_sd:.4f} sigma={sigma_sd:.4f}")

    subset_reco = np.abs(reco_resid - mu_reco) < sigma_reco
    subset_sd = np.abs(sd_resid - mu_sd) < sigma_sd
    print(f"subset A (reco within its own 1-sigma): {subset_reco.sum()} events "
          f"({100 * subset_reco.mean():.1f}%)")
    print(f"subset B (single-draw within its own 1-sigma): {subset_sd.sum()} events "
          f"({100 * subset_sd.mean():.1f}%)")
    print(f"overlap (both): {(subset_reco & subset_sd).sum()} events")

    ncols = 3
    nrows = int(np.ceil(len(OTHER_KEYS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for i, key in enumerate(OTHER_KEYS):
        ax = axes[i // ncols][i % ncols]
        vals_reco_subset = truth[key][subset_reco]
        vals_sd_subset = truth[key][subset_sd]
        bins = np.histogram_bin_edges(truth[key], bins=40)
        ax.hist(vals_reco_subset, bins=bins, density=True, histtype="step", linewidth=1.6,
                label=f"reco-selected (n={subset_reco.sum()})", color="C0")
        ax.hist(vals_sd_subset, bins=bins, density=True, histtype="step", linewidth=1.6,
                label=f"single-draw-selected (n={subset_sd.sum()})", color="C1")
        print(f"[{key}] truth, reco-selected: mean={vals_reco_subset.mean():.4f} "
              f"std={vals_reco_subset.std():.4f}  |  single-draw-selected: "
              f"mean={vals_sd_subset.mean():.4f} std={vals_sd_subset.std():.4f}")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel(key, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=7)

    for j in range(len(OTHER_KEYS), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Truth-level distributions: events where reco gets dphi_jj right "
                 f"(within its own fitted 1$\\sigma$={sigma_reco:.3f}) vs. events where "
                 f"the single-draw unfolding gets dphi_jj right "
                 f"(within its own fitted 1$\\sigma$={sigma_sd:.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
