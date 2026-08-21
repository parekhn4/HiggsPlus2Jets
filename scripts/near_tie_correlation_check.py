"""
Usage
    python scripts/near_tie_correlation_check.py --checkpoint runs/2026-07-10_no_energy_16b/model.pt \
        --config configs/no_energy.yaml --n-samples 50

Rerun of the 2026-08-11 near-pT-tie / dphi_jj-posterior-spread correlation check (see
CLAUDE.md "dphi_jj residual/loss discussion"), but on the FULL held-out validation fold
(pooled across all 3 scenarios, read directly from the checkpoint's own preprocessed.h5 --
same fold-membership guarantee as scripts/compare_unfolded_old_vs_new_truth.py) instead of
the original 9,000-event subsample from an unsaved scratch script. Tests whether the
original "real but weak" correlation was itself statistics-limited.

For each held-out event: relative truth pT gap |pt_j1-pt_j2|/(pt_j1+pt_j2), and the
circular std of --n-samples posterior draws of dphi_jj (pT-ordered, training convention)
sqrt(-2*ln(R)) where R is the mean resultant length. Reports the Pearson correlation,
a decile breakdown, and the near-tie (bottom 20% pT gap) enrichment among the top-10%-
spread events -- same three statistics the original note reported, for direct comparison.
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

    X_list, Y_list = [], []
    for scenario in SCENARIOS:
        df = pd.read_hdf(preprocessed_path, key=scenario)
        val_df = df[df["AUX_fold"] == args.val_fold]
        print(f"[{scenario}] val fold: {len(val_df)} events")
        X_list.append(val_df[X_COLS].to_numpy(dtype=np.float32))
        Y_list.append(val_df[Y_COLS].to_numpy(dtype=np.float32))
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    n_events = len(X)
    print(f"pooled val fold: {n_events} events, {args.n_samples} draws/event")

    truth_fv = kinematics.reconstruct_event(Y, resolved["truth"])
    truth_obs = kinematics.build_observables(truth_fv)
    j1_pt, j2_pt = truth_obs["j1_pt"], truth_obs["j2_pt"]
    pt_gap = np.abs(j1_pt - j2_pt) / (j1_pt + j2_pt)

    X_scaled = inference_prep.apply_reco_scaling(X, scaler)
    torch.manual_seed(args.seed)
    samples_scaled = sample_posterior_batch(
        model, X_scaled, truth_dim, n_samples_per_event=args.n_samples,
        device=device, batch_size=args.batch_size,
    )
    samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
    unfolded_fv = kinematics.reconstruct_event(samples, resolved["truth"])
    unfolded_obs = kinematics.build_observables(unfolded_fv)
    dphi = unfolded_obs["dphi"].reshape(n_events, args.n_samples)

    sin_mean = np.mean(np.sin(dphi), axis=1)
    cos_mean = np.mean(np.cos(dphi), axis=1)
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    circ_std = np.sqrt(np.clip(-2.0 * np.log(np.clip(R, 1e-12, 1.0)), 0, None))

    corr = float(np.corrcoef(pt_gap, circ_std)[0, 1])
    print(f"\nPearson corr(pt_gap, circ_std) = {corr:.4f}  (n={n_events})")

    deciles = pd.qcut(pt_gap, 10, labels=False)
    print("\nBinned by pT-gap decile (0=nearest-tie, 9=most-separated):")
    for d in range(10):
        mask = deciles == d
        print(f"  decile {d}: mean pt_gap={pt_gap[mask].mean():.4f}  mean circ_std={circ_std[mask].mean():.4f}  n={mask.sum()}")

    bottom20 = pt_gap <= np.percentile(pt_gap, 20)
    top10_spread = circ_std >= np.percentile(circ_std, 90)
    frac_bottom20_among_top10 = bottom20[top10_spread].mean()
    enrichment = frac_bottom20_among_top10 / 0.2
    print(f"\nNear-tie (bottom 20% pT gap) fraction among top-10%-spread events: "
          f"{frac_bottom20_among_top10:.4f}  (base rate 0.20, enrichment {enrichment:.2f}x)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
