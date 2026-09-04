"""
Usage
    python scripts/dphi_roughness_check.py --checkpoint runs/2026-07-10_no_energy_16b/model.pt \
        --config configs/no_energy.yaml --n-samples 100 --n-bins 80

Quantitative test for whether the unfolded dphi_jj posterior is genuinely "spiky" (real
fine-scale structure in what the model outputs) or whether apparent jaggedness in a
rendered plot is just ordinary sampling noise -- the question raised after
scripts/compare_unfolded_old_vs_new_truth.py's 100-draws run, instead of judging it by
eye from a PDF (a mistake documented earlier this session).

Method: bin the (pooled, held-out-val-fold) unfolded posterior into raw counts, and
compare the observed bin-to-bin second-difference variance against what independent-
Poisson counting noise alone would predict for a smoothly-varying true density --
Var(c[i+1] - 2c[i] + c[i-1]) = c[i+1] + 4c[i] + c[i-1] under that null. A ratio near 1
means the histogram's roughness is fully explained by finite-sample noise (smooth
underlying posterior); a ratio >> 1 means real structure beyond sampling noise.

Runs this for dphi_jj (pT-ordered), H_pt as a smooth control (same posterior, should
come back near 1 if the statistic itself is well-behaved), and old TRUTH's own dphi_jj
(decoded directly from preprocessed.h5, no model involved) as a "how rough is the real
physics itself" reference -- the ΔR_jj>0.4 cut is a real sharp feature, so some elevated
roughness in truth itself is expected and not evidence of a model problem.
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
import inference.preprocessing_inference as inference_prep
from inference.inference import load_checkpoint_bundle, sample_posterior_batch

SCENARIOS = ["at_0_bt_1", "at_1_bt_0", "at_1_bt_1"]
X_COLS = [f"x_{i}" for i in range(66)]
Y_COLS = [f"y_{i}" for i in range(12)]


def roughness_ratio(values: np.ndarray, bins: np.ndarray) -> tuple[float, float, float]:
    counts, _ = np.histogram(values, bins=bins)
    c = counts.astype(np.float64)
    second_diff = c[2:] - 2 * c[1:-1] + c[:-2]
    observed = np.sum(second_diff ** 2)
    expected = np.sum(c[2:] + 4 * c[1:-1] + c[:-2])
    return observed, expected, observed / expected


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    model, resolved, max_jets, scaler = bundle["model"], bundle["resolved"], bundle["max_jets"], bundle["scaler"]
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)
    preprocessed_path = Path(args.checkpoint).parent / "preprocessed.h5"
    print(f"checkpoint: epoch {bundle['epoch']}, val_loss {bundle['val_loss']:.4f}")

    X_list, Y_list = [], []
    for scenario in SCENARIOS:
        df = pd.read_hdf(preprocessed_path, key=scenario)
        val_df = df[df["AUX_fold"] == args.val_fold]
        X_list.append(val_df[X_COLS].to_numpy(dtype=np.float32))
        Y_list.append(val_df[Y_COLS].to_numpy(dtype=np.float32))
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    n_events = len(X)
    print(f"pooled val fold: {n_events} events, {args.n_samples} draws/event")

    truth_obs = kinematics.build_observables(kinematics.reconstruct_event(Y, resolved["truth"]))

    X_scaled = inference_prep.apply_reco_scaling(X, scaler)
    torch.manual_seed(args.seed)
    samples_scaled = sample_posterior_batch(
        model, X_scaled, truth_dim, n_samples_per_event=args.n_samples,
        device=device, batch_size=args.batch_size,
    )
    samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
    unfolded_obs = kinematics.build_observables(kinematics.reconstruct_event(samples, resolved["truth"]))

    dphi_bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)
    hpt_bins = np.linspace(0, 400, args.n_bins + 1)

    print(f"\n{'series':35s} {'observed':>14s} {'expected':>14s} {'ratio':>8s}")
    for label, values, bins in [
        ("unfolded dphi_jj (pT-ordered)", unfolded_obs["dphi"], dphi_bins),
        ("unfolded H_pt (smooth control)", unfolded_obs["H_pt"], hpt_bins),
        ("old truth dphi_jj (no model)", truth_obs["dphi"], dphi_bins),
    ]:
        obs, exp, ratio = roughness_ratio(values, bins)
        print(f"{label:35s} {obs:14.1f} {exp:14.1f} {ratio:8.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-bins", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
