"""
Usage
    python scripts/compare_unfolded_old_vs_new_truth.py \
        --checkpoint runs/2026-07-10_no_energy_16b/model.pt \
        --config configs/no_energy.yaml --data-dir fromPratik_sim_delphes/ \
        --mjj-cut 100.0 --output-dir ../pratik_eval/unfolded_old_vs_new_truth_baseline/ \
        --n-samples 100 --bin-multiplier 4

Compares the OLD checkpoint's UNFOLDED posterior (many draws/event, pooled -- not a
single-draw reduction) on its own genuine held-out validation fold (read directly from
the checkpoint's own preprocessed.h5, AUX_fold == val_fold -- guarantees no train
contamination and exact correspondence to what the checkpoint actually trained on,
unlike re-deriving from ROOT with the current config) against the NEW data's actual
parton-level (truth) observables, with the hard-scatter m_jj > --mjj-cut cut reinstated
(see scripts/check_mjj_cut_effect.py / CLAUDE.md). Covers all 10 build_observables()
keys, not just dphi_jj -- one output file per scenario + pooled, same fine binning
(scripts/closure_pratik_mjj_cut.py's finer_plot_specs) applied to both curves in every
panel.

n_samples_per_event defaults to 100 (not count-matched to the new-truth sample size --
density-normalized histograms don't need equal counts, just enough in each to resolve
the chosen binning without being noise-dominated).
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
import inference.preprocessing_inference as inference_prep
from inference.inference import load_checkpoint_bundle, sample_posterior_batch
from scripts.evaluate_pratik_data import read_and_select, SCENARIOS
from scripts.check_mjj_cut_effect import invariant_mass
from scripts.closure_pratik_mjj_cut import finer_plot_specs

X_COLS = [f"x_{i}" for i in range(66)]
OLD_COLOR = "C0"
NEW_COLOR = "C1"
OBS_KEYS = ["H_pt", "H_eta", "H_phi", "j1_pt", "j1_eta", "j2_pt", "j2_eta", "dphi", "dphi_eta_ordered", "deta"]


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    model, resolved, max_jets, scaler = bundle["model"], bundle["resolved"], bundle["max_jets"], bundle["scaler"]
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)
    val_fold = args.val_fold

    preprocessed_path = Path(args.checkpoint).parent / "preprocessed.h5"
    print(f"checkpoint: epoch {bundle['epoch']}, val_loss {bundle['val_loss']:.4f}, "
          f"val_fold {val_fold}, preprocessed: {preprocessed_path}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_specs = finer_plot_specs(args.bin_multiplier)

    # 1) new data: parton-level truth, all observables, with the m_jj cut
    new_cut = {}
    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        print(f"[{scenario}] reading {path}")
        _, truth_fv = read_and_select(str(path), config)
        obs = kinematics.build_observables(truth_fv)
        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        new_cut[scenario] = {k: obs[k][keep] for k in OBS_KEYS}
        print(f"  {len(obs['dphi'])} events, {keep.sum()} after m_jj > {args.mjj_cut} GeV")

    # 2) old data: held-out val fold only, from the checkpoint's own preprocessed.h5
    n_val = {}
    x_val = {}
    for scenario in SCENARIOS:
        df = pd.read_hdf(preprocessed_path, key=scenario)
        val_df = df[df["AUX_fold"] == val_fold]
        n_val[scenario] = len(val_df)
        x_val[scenario] = val_df[X_COLS].to_numpy(dtype=np.float32)
        print(f"[{scenario}] val fold: {n_val[scenario]} events")

    n_samples = args.n_samples
    print(f"\nn_samples_per_event = {n_samples}")
    for s in SCENARIOS:
        print(f"  {s}: {n_val[s]} val events x {n_samples} = {n_val[s] * n_samples} unfolded draws")

    # 3) unfold the old val-fold events, n_samples draws/event, pooled
    unfolded = {}
    for scenario in SCENARIOS:
        print(f"\n[{scenario}] unfolding {n_val[scenario]} val events x {n_samples} samples")
        X_reco_scaled = inference_prep.apply_reco_scaling(x_val[scenario], scaler)
        torch.manual_seed(args.seed)
        samples_scaled = sample_posterior_batch(
            model, X_reco_scaled, truth_dim, n_samples_per_event=n_samples,
            device=device, batch_size=args.batch_size,
        )
        samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
        unfolded_fv = kinematics.reconstruct_event(samples, resolved["truth"])
        obs = kinematics.build_observables(unfolded_fv)
        unfolded[scenario] = {k: obs[k] for k in OBS_KEYS}

    pooled_new = {k: np.concatenate([new_cut[s][k] for s in SCENARIOS]) for k in OBS_KEYS}
    pooled_unfolded = {k: np.concatenate([unfolded[s][k] for s in SCENARIOS]) for k in OBS_KEYS}

    rows = SCENARIOS + ["pooled"]
    data_new = {**new_cut, "pooled": pooled_new}
    data_unfolded = {**unfolded, "pooled": pooled_unfolded}

    print()
    for row_label in rows:
        write_plots(data_new[row_label], data_unfolded[row_label], plot_specs,
                    args.output_dir, row_label, n_samples, args.mjj_cut)
    print(f"\nDone. Plots under {args.output_dir}/")


def write_plots(new_obs: dict, unfolded_obs: dict, plot_specs: list, output_dir: str,
                 label: str, n_samples: int, mjj_cut: float) -> None:
    ncols = 3
    nrows = int(np.ceil(len(plot_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, (key, bins, xlabel, subtitle) in enumerate(plot_specs):
        ax = axes[i // ncols][i % ncols]
        new_arr, unf_arr = new_obs[key], unfolded_obs[key]
        new_h, _ = np.histogram(new_arr, bins=bins, density=True)
        unf_h, _ = np.histogram(unf_arr, bins=bins, density=True)
        sse = float(np.sum((new_h - unf_h) ** 2))
        print(f"[{label}] {key}: SSE(unfolded-old vs new-truth-cut) = {sse:.6f} "
              f"(n_new={len(new_arr)}, n_unfolded={len(unf_arr)})")

        ax.hist(new_arr, bins=bins, density=True, histtype="step", linewidth=1.6,
                color=NEW_COLOR, label=f"new, parton truth, $m_{{jj}}$>{mjj_cut:.0f} GeV")
        ax.hist(unf_arr, bins=bins, density=True, histtype="step", linewidth=1.6,
                color=OLD_COLOR, label=f"old, unfolded ({n_samples} draws/event)")
        ax.set_title(subtitle, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=6)

    for j in range(len(plot_specs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Unfolded old ({n_samples} draws/event) vs. new parton truth "
                 f"($m_{{jj}}$>{mjj_cut:.0f} GeV): {label}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"unfolded_old_vs_new_{label}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mjj-cut", type=float, default=100.0)
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bin-multiplier", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
