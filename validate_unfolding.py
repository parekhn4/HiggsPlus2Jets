"""
Usage
    python validate_unfolding.py \\
        --checkpoint best_model.pt \\
        --config configs/no_energy.yaml \\
        --preprocessed preprocessed.h5 \\
        --n-samples 200 \\
        --output-dir validation_plots/

Compares three ways of collapsing the posterior into a per-event unfolded
value: the on-shell mean (kinematics.average_posterior_samples), a single
random posterior draw per event (kinematics.select_posterior_draw -- no
reweighting needed, every draw is already unweighted/exact), and the full
pooled posterior (kinematics.reconstruct_event, every sample kept) --
against truth and reco, on the checkpoint's own held-out validation fold
(unlike inference.py's real-data path, truth is available here). Writes,
per scenario (plus pooled, all scenarios combined):

    closure_mean_{scenario}.pdf        truth/reco/unfolded-mean marginal shapes
    closure_single_draw_{scenario}.pdf truth/reco/unfolded-single-draw marginal shapes
    closure_samples_{scenario}.pdf     truth/reco/unfolded-all-samples marginal shapes
    error_hist_{scenario}.pdf          per-event (truth - X) residuals, X in
                                        {reco, mean, single draw, all samples},
                                        overlaid so you can see which reduction
                                        strategy sits tighter around zero
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import kinematics
import plotting
from evaluate import load_val_fold
from inference import load_checkpoint_bundle, sample_posterior_batch
import preprocessing_inference as inference_prep


def validate_scenario(scenario: str, df_val: pd.DataFrame, bundle: dict,
                       n_samples: int, device: str, batch_size: int) -> dict:
    resolved, max_jets, scaler, model = (
        bundle["resolved"], bundle["max_jets"], bundle["scaler"], bundle["model"]
    )
    reco_dim = kinematics.total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)

    x_cols = [f"x_{i}" for i in range(reco_dim)]
    y_cols = [f"y_{i}" for i in range(truth_dim)]
    X_reco = df_val[x_cols].to_numpy(dtype=np.float32)
    y_truth = df_val[y_cols].to_numpy(dtype=np.float32)
    n_events = len(X_reco)

    truth_fv = kinematics.reconstruct_event(y_truth, resolved["truth"])
    reco_fv = kinematics.reco_four_vectors(X_reco, resolved["reco"], max_jets)

    X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)
    samples_scaled = sample_posterior_batch(
        model, X_reco_scaled, truth_dim, n_samples_per_event=n_samples,
        device=device, batch_size=batch_size,
    )
    samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)

    unfolded_mean_fv = kinematics.average_posterior_samples(
        samples, resolved["truth"], n_events=n_events, n_samples=n_samples
    )

    single_draw = kinematics.select_posterior_draw(samples, n_events, n_samples, draw_index=0)
    unfolded_single_draw_fv = kinematics.reconstruct_event(single_draw, resolved["truth"])

    # (n_events*n_samples, 4) -> (n_events, n_samples, 4) per object, so
    # observables built from this come out shaped (n_events, n_samples) --
    # what observable_residual/plot_error_histograms expect for a full posterior
    unfolded_samples_fv = kinematics.reconstruct_event(samples, resolved["truth"])
    unfolded_samples_fv = {name: fv.reshape(n_events, n_samples, 4) for name, fv in unfolded_samples_fv.items()}

    return {
        "truth": kinematics.build_observables(truth_fv),
        "reco": kinematics.build_observables(reco_fv),
        "unfolded_mean": kinematics.build_observables(unfolded_mean_fv),
        "unfolded_single_draw": kinematics.build_observables(unfolded_single_draw_fv),
        "unfolded_samples": kinematics.build_observables(unfolded_samples_fv),
    }


def write_plots(result: dict, plot_specs: list, error_specs: list,
                 output_dir: str, title_suffix: str, file_suffix: str) -> None:
    fig = plotting.plot_closure(
        result["truth"], result["reco"], result["unfolded_mean"],
        plot_specs=plot_specs, title=f"Closure (mean unfolding): {title_suffix}",
    )
    out_path = Path(output_dir) / f"closure_mean_{file_suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")

    fig = plotting.plot_closure(
        result["truth"], result["reco"], result["unfolded_single_draw"],
        plot_specs=plot_specs, title=f"Closure (single random draw): {title_suffix}",
    )
    out_path = Path(output_dir) / f"closure_single_draw_{file_suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")

    fig = plotting.plot_closure(
        result["truth"], result["reco"], result["unfolded_samples"],
        plot_specs=plot_specs, title=f"Closure (full posterior): {title_suffix}",
    )
    out_path = Path(output_dir) / f"closure_samples_{file_suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")

    fig = plotting.plot_error_histograms(
        result["truth"],
        [
            (result["reco"], "reco"),
            (result["unfolded_mean"], "unfolded (mean)"),
            (result["unfolded_single_draw"], "unfolded (single draw)"),
            (result["unfolded_samples"], "unfolded (all samples)"),
        ],
        reference_label="truth", error_specs=error_specs, title=f"Residuals vs truth: {title_suffix}",
    )
    out_path = Path(output_dir) / f"error_hist_{file_suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")


def run_validate(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    val_fold = torch.load(args.checkpoint, map_location=device, weights_only=False)["val_fold"]
    print(f"validating on held-out fold {val_fold} (the checkpoint's own validation fold)")

    reco_dim = kinematics.total_dim(bundle["resolved"]["reco"], max_jets=bundle["max_jets"])
    truth_dim = kinematics.total_dim(bundle["resolved"]["truth"], max_jets=bundle["max_jets"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with pd.HDFStore(args.preprocessed, mode="r") as store:
        scenarios = [k.strip("/") for k in store.keys()]

    plot_specs = list(plotting.DEFAULT_PLOT_SPECS) + [
        ("dphi_eta_ordered", np.linspace(-np.pi, np.pi, 20),
         r"$\Delta\phi_{jj}$ (eta-ordered, CP)", r"$\Delta\phi_{jj}$ (CP convention)"),
    ]
    error_specs = list(plotting.DEFAULT_ERROR_SPECS) + [
        ("dphi_eta_ordered", np.linspace(-1.0, 1.0, 41), r"$\Delta\phi_{jj}$ (eta-ordered, CP) residual"),
    ]

    pooled = {"truth": [], "reco": [], "unfolded_mean": [], "unfolded_single_draw": [], "unfolded_samples": []}

    for scenario in scenarios:
        print(f"\n[{scenario}]")
        df_val = load_val_fold(args.preprocessed, scenario, val_fold, reco_dim, truth_dim)
        print(f"  {len(df_val)} held-out events")
        if len(df_val) == 0:
            print("  skipping -- no events in this fold for this scenario")
            continue

        result = validate_scenario(scenario, df_val, bundle, args.n_samples, device, args.batch_size)
        write_plots(result, plot_specs, error_specs, args.output_dir, f"{scenario} (fold {val_fold})", scenario)

        for key in pooled:
            pooled[key].append(result[key])

    if not any(pooled["truth"]):
        print("\nno scenarios had held-out events -- skipping pooled plot")
        return

    print(f"\n[pooled, all {len(scenarios)} scenarios combined]")
    pooled_obs = {}
    for domain, obs_dicts in pooled.items():
        pooled_obs[domain] = {
            key: np.concatenate([d[key] for d in obs_dicts])
            for key in obs_dicts[0].keys()
        }
    write_plots(pooled_obs, plot_specs, error_specs, args.output_dir,
                f"pooled, all scenarios (fold {val_fold})", "pooled")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare mean-vs-full-posterior unfolding against truth/reco on the held-out fold."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--preprocessed", required=True, help="preprocessed.h5 from preprocessing_training.py")
    p.add_argument("--output-dir", default="validation_plots")
    p.add_argument("--n-samples", type=int, default=200,
                    help="Posterior samples per event (default: 200)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42, help="Seed for posterior sampling (default: 42)")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_validate(args)


if __name__ == "__main__":
    main()
