"""
Usage
    python scripts/old_data_stats_visibility_check.py --checkpoint runs/2026-07-10_no_energy_16b/model.pt \
        --config configs/no_energy.yaml --n-samples 36 --output ../pratik_eval/old_data_stats_visibility.pdf

Purely-internal-to-old-data check: does the j2_pt (and j1_pt, dphi, for contrast) closure
wobble seen on the new (Pratik) data already exist within the OLD data's own held-out
validation fold, just hidden by its small sample size (~17.6k events/scenario) at the
historical single-draw closure -- or does it only appear once new data (and whatever
distribution differences it carries, e.g. the m_jj cut being a coarse post-hoc match
rather than a true generation-level one) enters the picture?

No new data involved at all here -- three curves per observable, same held-out old val-fold
events throughout: (1) truth, (2) unfolded with n_samples=1 (matching what the historical
single-draw closure plots would show, same statistics as evaluate.py/validate_unfolding.py
originally used), (3) unfolded with --n-samples (default 36, pooled across draws) -- same
events, same model, only the number of posterior draws differs. If (3) shows structure
that (2) doesn't, the wobble was there all along and just invisible at the original
statistics. If (2) and (3) both look equally clean, the new-data-specific wobble is real
and not a statistics artifact.
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

import core.kinematics as kinematics
import plotting.plotting as plotting
import inference.preprocessing_inference as inference_prep
from inference.inference import load_checkpoint_bundle, sample_posterior_batch

SCENARIOS = ["at_0_bt_1", "at_1_bt_0", "at_1_bt_1"]
X_COLS = [f"x_{i}" for i in range(66)]
Y_COLS = [f"y_{i}" for i in range(12)]

OBS_SPECS = [
    ("j2_pt", np.linspace(0, 300, 31), r"$p_{T,j2}$ [GeV]"),
    ("j1_pt", np.linspace(0, 300, 31), r"$p_{T,j1}$ [GeV]"),
    ("dphi", np.linspace(-np.pi, np.pi, 20), r"$\Delta\phi_{jj}$"),
]


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    model, resolved, max_jets, scaler = bundle["model"], bundle["resolved"], bundle["max_jets"], bundle["scaler"]
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)
    preprocessed_path = Path(args.checkpoint).parent / "preprocessed.h5"
    print(f"checkpoint: epoch {bundle['epoch']}, val_loss {bundle['val_loss']:.4f}, "
          f"preprocessed: {preprocessed_path}")

    per_scenario = {}
    for scenario in SCENARIOS:
        df = pd.read_hdf(preprocessed_path, key=scenario)
        val_df = df[df["AUX_fold"] == args.val_fold]
        n_val = len(val_df)
        print(f"[{scenario}] val fold: {n_val} events")
        X = val_df[X_COLS].to_numpy(dtype=np.float32)
        Y = val_df[Y_COLS].to_numpy(dtype=np.float32)

        truth_fv = kinematics.reconstruct_event(Y, resolved["truth"])
        truth_obs = kinematics.build_observables(truth_fv)

        X_scaled = inference_prep.apply_reco_scaling(X, scaler)

        torch.manual_seed(args.seed)
        single_scaled = sample_posterior_batch(
            model, X_scaled, truth_dim, n_samples_per_event=1,
            device=device, batch_size=args.batch_size,
        )
        single = inference_prep.invert_truth_scaling(single_scaled, scaler)
        single_obs = kinematics.build_observables(kinematics.reconstruct_event(single, resolved["truth"]))

        torch.manual_seed(args.seed)
        many_scaled = sample_posterior_batch(
            model, X_scaled, truth_dim, n_samples_per_event=args.n_samples,
            device=device, batch_size=args.batch_size,
        )
        many = inference_prep.invert_truth_scaling(many_scaled, scaler)
        many_obs = kinematics.build_observables(kinematics.reconstruct_event(many, resolved["truth"]))

        per_scenario[scenario] = {"truth": truth_obs, "single": single_obs, "many": many_obs}

    pooled = {
        domain: {key: np.concatenate([per_scenario[s][domain][key] for s in SCENARIOS])
                 for key in per_scenario[SCENARIOS[0]][domain].keys()}
        for domain in ["truth", "single", "many"]
    }
    rows = SCENARIOS + ["pooled"]
    data = {**per_scenario, "pooled": pooled}

    fig, axes = plt.subplots(len(rows), len(OBS_SPECS), figsize=(6 * len(OBS_SPECS), 4.2 * len(rows)))
    print()
    for i, row_label in enumerate(rows):
        d = data[row_label]
        for j, (key, bins, xlabel) in enumerate(OBS_SPECS):
            truth_h, _ = np.histogram(d["truth"][key], bins=bins, density=True)
            single_h, _ = np.histogram(d["single"][key], bins=bins, density=True)
            many_h, _ = np.histogram(d["many"][key], bins=bins, density=True)
            sse_single = float(np.sum((single_h - truth_h) ** 2))
            sse_many = float(np.sum((many_h - truth_h) ** 2))
            print(f"[{row_label}] {key}: SSE(single-draw vs truth)={sse_single:.5f}  "
                  f"SSE({args.n_samples}-draws vs truth)={sse_many:.5f}")

            ax = axes[i][j]
            ax.hist(d["truth"][key], bins=bins, density=True, histtype="step", linewidth=1.8,
                    color="0.2", label="truth")
            ax.hist(d["single"][key], bins=bins, density=True, histtype="step", linewidth=1.5,
                    color="C0", label="unfolded, single-draw")
            ax.hist(d["many"][key], bins=bins, density=True, histtype="step", linewidth=1.5,
                    color="C1", label=f"unfolded, {args.n_samples} draws")
            ax.set_title(f"{row_label} -- {xlabel}", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.legend(fontsize=7)

    fig.suptitle("Old-data-only check: does more posterior sampling reveal structure hidden "
                 "at single-draw statistics?", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=36)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
