from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import kinematics
import plotting
from inference import load_checkpoint_bundle, sample_posterior_batch
import preprocessing_inference as inference_prep


def load_val_fold(preprocessed_path: str, scenario: str, val_fold: int,
                   reco_dim: int, truth_dim: int) -> pd.DataFrame:
    with pd.HDFStore(preprocessed_path, mode="r") as store:
        df = store[scenario]
    return df[df["AUX_fold"] == val_fold].reset_index(drop=True)


def evaluate_scenario(scenario: str, df_val: pd.DataFrame, bundle: dict,
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

    # ── truth four-vectors: decode directly, no model involved ──
    truth_fv = kinematics.reconstruct_event(y_truth, resolved["truth"])
    truth_obs = kinematics.build_observables(truth_fv)

    # ── reco four-vectors: H from photons, j1/j2 from the first two
    #    pT-ordered jet slots as a "detector-level j1/j2" proxy (jets are
    #    pT-ordered in the encoded reco array regardless of parton_ordering,
    #    which only affects truth) ──
    reco_fv = kinematics.reco_four_vectors(X_reco, resolved["reco"], max_jets)
    reco_obs = kinematics.build_observables(reco_fv)

    # ── unfolded (flow) four-vectors: sample posterior, one draw/event
    #    for closure comparison (matches sample count of truth/reco) ──
    X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)
    samples_scaled = sample_posterior_batch(
        model, X_reco_scaled, truth_dim, n_samples_per_event=n_samples,
        device=device, batch_size=batch_size,
    )
    samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
    flow_fv = kinematics.reconstruct_event(samples, resolved["truth"])
    flow_obs = kinematics.build_observables(flow_fv)

    return {"truth": truth_obs, "reco": reco_obs, "flow": flow_obs}


def run_evaluate(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    val_fold = torch.load(args.checkpoint, map_location=device, weights_only=False)["val_fold"]
    print(f"evaluating on held-out fold {val_fold} (the checkpoint's own validation fold)")

    reco_dim = kinematics.total_dim(bundle["resolved"]["reco"], max_jets=bundle["max_jets"])
    truth_dim = kinematics.total_dim(bundle["resolved"]["truth"], max_jets=bundle["max_jets"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with pd.HDFStore(args.preprocessed, mode="r") as store:
        scenarios = [k.strip("/") for k in store.keys()]

    plot_specs = list(plotting.DEFAULT_PLOT_SPECS) + [
        ("dphi_eta_ordered", np.linspace(-np.pi, np.pi, 20),
         r"$\Delta\phi_{jj}$ (eta-ordered, CP)", r"$\Delta\phi_{jj}$ (CP convention)"),
    ]

    # accumulated across all scenarios, for the pooled plot -- this is the
    # closest proxy to "real data," which won't be separable by CP scenario
    pooled = {"truth": [], "reco": [], "flow": []}

    for scenario in scenarios:
        print(f"\n[{scenario}]")
        df_val = load_val_fold(args.preprocessed, scenario, val_fold, reco_dim, truth_dim)
        print(f"  {len(df_val)} held-out events")
        if len(df_val) == 0:
            print("  skipping -- no events in this fold for this scenario")
            continue

        result = evaluate_scenario(scenario, df_val, bundle, args.n_samples, device, args.batch_size)

        fig = plotting.plot_closure(
            result["truth"], result["reco"], result["flow"],
            plot_specs=plot_specs, title=f"Closure: {scenario} (fold {val_fold})",
        )
        out_path = Path(args.output_dir) / f"closure_{scenario}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"  wrote {out_path}")

        for key in ("truth", "reco", "flow"):
            pooled[key].append(result[key])

    if not any(pooled["truth"]):
        print("\nno scenarios had held-out events -- skipping pooled plot")
        return

    # ── pooled plot: all scenarios combined, no CP-scenario distinction --
    # this is the closure test that best matches what evaluating on real
    # data will actually look like, since real events won't come pre-labeled
    # by CP coupling point ──
    print(f"\n[pooled, all {len(scenarios)} scenarios combined]")
    pooled_obs = {}
    for domain in ("truth", "reco", "flow"):
        obs_dicts = pooled[domain]
        pooled_obs[domain] = {
            key: np.concatenate([d[key] for d in obs_dicts])
            for key in obs_dicts[0].keys()
        }

    fig = plotting.plot_closure(
        pooled_obs["truth"], pooled_obs["reco"], pooled_obs["flow"],
        plot_specs=plot_specs, title=f"Closure: pooled, all scenarios (fold {val_fold})",
    )
    out_path = Path(args.output_dir) / "closure_pooled.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Closure-test a trained checkpoint on its held-out fold.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--preprocessed", required=True, help="preprocessed.h5 from preprocessing_training.py")
    p.add_argument("--output-dir", default="eval_plots")
    p.add_argument("--n-samples", type=int, default=200,
                    help="Posterior samples per event for closure comparison (default: 200)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42, help="Seed for posterior sampling (default: 42)")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_evaluate(args)


if __name__ == "__main__":
    main()