"""
Usage
    python scripts/error_hist_reco_vs_singledraw.py \
        --checkpoint runs/2026-08-21_no_energy_pratik_16b/model_epoch268_checkpoint.pt \
        --config configs/no_energy_pratik.yaml \
        --preprocessed runs/2026-08-21_no_energy_pratik_16b/preprocessed.h5 \
        --output runs/2026-08-21_no_energy_pratik_16b/validation_plots_epoch268_fine/error_hist_pooled_reco_vs_singledraw.pdf \
        --bin-multiplier 4

Pared-down version of scripts/validate_unfolding.py's error_hist_pooled plot -- same
pooled (all-scenario) held-out val fold, same residual definition, but only
`reco` and `unfolded (single draw)` overlaid (drops mean/full-posterior/z=0), since a
single n_samples_per_event=1 draw is already exactly the "single draw" reduction --
no need to run the full mean/samples/zero machinery just to get this one comparison.
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

    truth_list, reco_list, single_draw_list = [], [], []
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
        single_draw_fv = kinematics.reconstruct_event(samples, resolved["truth"])

        truth_list.append(kinematics.build_observables(truth_fv))
        reco_list.append(kinematics.build_observables(reco_fv))
        single_draw_list.append(kinematics.build_observables(single_draw_fv))

    pooled_truth = {k: np.concatenate([d[k] for d in truth_list]) for k in truth_list[0]}
    pooled_reco = {k: np.concatenate([d[k] for d in reco_list]) for k in reco_list[0]}
    pooled_single_draw = {k: np.concatenate([d[k] for d in single_draw_list]) for k in single_draw_list[0]}

    error_specs = list(plotting.DEFAULT_ERROR_SPECS) + [
        ("dphi_eta_ordered", np.linspace(-1.0, 1.0, 41), r"$\Delta\phi_{jj}$ (eta-ordered, CP) residual"),
    ]
    m = args.bin_multiplier
    if m != 1:
        error_specs = [(key, np.linspace(bins[0], bins[-1], (len(bins) - 1) * m + 1), xlabel)
                       for key, bins, xlabel in error_specs]
    available_keys = kinematics.available_observable_keys(resolved["truth"]["objects"])
    error_specs = plotting.filter_specs(error_specs, available_keys)

    comparisons = [
        (pooled_reco, "reco"),
        (pooled_single_draw, "unfolded (single draw)"),
    ]
    norm_str = "raw counts" if args.no_density else "density-normalized"
    fig = plotting.plot_error_histograms(
        pooled_truth, comparisons, reference_label="truth", error_specs=error_specs,
        title=f"Residuals vs truth (reco + single-draw only, {norm_str}): pooled (fold {val_fold})",
        density=not args.no_density,
    )
    fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bin-multiplier", type=int, default=1)
    p.add_argument("--no-density", action="store_true",
                    help="Plot raw counts instead of density-normalizing each curve -- "
                         "valid here since reco and single-draw share the same "
                         "underlying event count (one residual/event each).")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
