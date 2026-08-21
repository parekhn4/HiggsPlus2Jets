"""
Usage
    python scripts/compare_unfolded_old_vs_new_truth.py \
        --checkpoint runs/2026-07-10_no_energy_16b/model.pt \
        --config configs/no_energy.yaml --data-dir fromPratik_sim_delphes/ \
        --mjj-cut 100.0 --output ../pratik_eval/unfolded_old_vs_new_truth_baseline.pdf \
        --n-bins 200

Compares the OLD checkpoint's UNFOLDED posterior (many draws/event, pooled -- not a
single-draw reduction) on its own genuine held-out validation fold (read directly from
the checkpoint's own preprocessed.h5, AUX_fold == val_fold -- guarantees no train
contamination and exact correspondence to what the checkpoint actually trained on,
unlike re-deriving from ROOT with the current config) against the NEW data's actual
parton-level (truth) dphi_jj, with the hard-scatter m_jj > --mjj-cut cut reinstated
(see scripts/check_mjj_cut_effect.py / CLAUDE.md).

n_samples_per_event is chosen (once, applied to every scenario) so the pooled unfolded
old-data draw count lands close to the new data's post-cut truth event count -- i.e.
val_fold_size * n_samples ~= new_cut_count -- so the two distributions being compared
carry comparable statistics, not an apples-to-oranges few-thousand-vs-hundreds-of-
thousands comparison.
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

X_COLS = [f"x_{i}" for i in range(66)]
OLD_COLOR = "C0"
NEW_COLOR = "C1"


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

    # 1) new data: parton-level truth dphi_jj with the m_jj cut, per scenario
    new_cut = {}
    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        print(f"[{scenario}] reading {path}")
        _, truth_fv = read_and_select(str(path), config)
        obs = kinematics.build_observables(truth_fv)
        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        new_cut[scenario] = {"dphi": obs["dphi"][keep], "dphi_eta": obs["dphi_eta_ordered"][keep]}
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

    # 3) pick one common n_samples so unfolded-old counts land close to new-cut counts
    if args.n_samples is not None:
        n_samples = args.n_samples
    else:
        ratios = [len(new_cut[s]["dphi"]) / n_val[s] for s in SCENARIOS]
        n_samples = int(round(np.mean(ratios)))
    print(f"\nn_samples_per_event = {n_samples}")
    for s in SCENARIOS:
        print(f"  {s}: {n_val[s]} val events x {n_samples} = {n_val[s] * n_samples} unfolded draws "
              f"(target {len(new_cut[s]['dphi'])})")

    # 4) unfold the old val-fold events, n_samples draws/event, pooled
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
        unfolded[scenario] = {"dphi": obs["dphi"], "dphi_eta": obs["dphi_eta_ordered"]}

    pooled_new = {k: np.concatenate([new_cut[s][k] for s in SCENARIOS]) for k in ["dphi", "dphi_eta"]}
    pooled_unfolded = {k: np.concatenate([unfolded[s][k] for s in SCENARIOS]) for k in ["dphi", "dphi_eta"]}

    rows = SCENARIOS + ["pooled"]
    data_new = {**new_cut, "pooled": pooled_new}
    data_unfolded = {**unfolded, "pooled": pooled_unfolded}

    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 4.2 * len(rows)))
    print()
    for i, row_label in enumerate(rows):
        for j, (obs_key, title) in enumerate([
            ("dphi", "pT-ordered (training convention)"),
            ("dphi_eta", r"$\eta$-ordered (CP convention)"),
        ]):
            new_arr = data_new[row_label][obs_key]
            unf_arr = data_unfolded[row_label][obs_key]
            new_h, _ = np.histogram(new_arr, bins=bins, density=True)
            unf_h, _ = np.histogram(unf_arr, bins=bins, density=True)
            sse = float(np.sum((new_h - unf_h) ** 2))
            print(f"[{row_label}] {obs_key}: SSE(unfolded-old vs new-truth-cut) = {sse:.5f} "
                  f"(n_new={len(new_arr)}, n_unfolded={len(unf_arr)})")

            ax = axes[i][j]
            ax.hist(new_arr, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color=NEW_COLOR, label=f"new, parton truth, $m_{{jj}}$>{args.mjj_cut:.0f} GeV")
            ax.hist(unf_arr, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color=OLD_COLOR, label=f"old, unfolded (val fold, {n_samples} draws/event)")
            ax.set_title(f"{row_label} -- {title}", fontsize=10)
            ax.set_xlabel(r"$\Delta\phi_{jj}$")
            ax.set_ylabel("density")
            ax.set_xlim(-np.pi, np.pi)
            ax.legend(fontsize=7)

    fig.suptitle(f"Unfolded old (val fold, {n_samples} draws/event) vs. new parton truth "
                 f"($m_{{jj}}$>{args.mjj_cut:.0f} GeV)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mjj-cut", type=float, default=100.0)
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=None,
                    help="Override the auto-computed common n_samples_per_event")
    p.add_argument("--output", required=True)
    p.add_argument("--n-bins", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
