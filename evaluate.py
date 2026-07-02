"""
evaluate.py — closure testing on a held-out fold: sample the posterior
for events the model never trained on, compare truth / reco / unfolded
distributions, per scenario.

Uses the SAME fold the checkpoint recorded as its validation fold
(checkpoint["val_fold"]) -- this is a genuine held-out closure test, not
just re-running on arbitrary data. Requires preprocessing_training.py's
output (has truth), not preprocessing_inference.py's -- closure testing
needs truth to compare against, unlike the plain inference path.

Reports both:
  - dphi_jj as encoded during training (whatever parton_ordering the
    config used -- pt or eta)
  - the CP-quality, literature-convention eta-ordered Delta phi_jj,
    re-derived directly from four-vectors regardless of training's
    parton_ordering choice (see kinematics.eta_ordered_dphi_jj)

Usage
-----
    python evaluate.py \\
        --checkpoint best_model.pt \\
        --config configs/no_energy.yaml \\
        --preprocessed preprocessed.h5 \\
        --n-samples 200 \\
        --output-dir eval_plots/
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
from inference import load_checkpoint_bundle, sample_posterior_batch
import preprocessing_inference as inference_prep


def load_val_fold(preprocessed_path: str, scenario: str, val_fold: int,
                   reco_dim: int, truth_dim: int) -> pd.DataFrame:
    with pd.HDFStore(preprocessed_path, mode="r") as store:
        df = store[scenario]
    return df[df["fold"] == val_fold].reset_index(drop=True)


def build_observables(four_vectors: dict) -> dict:
    """
    Build the plotting.py observable dict from a {"H":..,"j1":..,"j2":..}
    four-vector set. Includes both the as-labeled dphi_jj (whatever j1/j2
    already are) and the re-derived, literature-convention eta-ordered
    version, which is invariant to how j1/j2 got their labels.
    """
    H, j1, j2 = four_vectors["H"], four_vectors["j1"], four_vectors["j2"]

    def pt_eta_phi(fv):
        px, py = fv[..., 1], fv[..., 2]
        pt = np.sqrt(px ** 2 + py ** 2)
        eta = kinematics.four_vector_eta(fv)
        phi = kinematics.four_vector_phi(fv)
        return pt, eta, phi

    H_pt, H_eta, H_phi = pt_eta_phi(H)
    j1_pt, j1_eta, j1_phi = pt_eta_phi(j1)
    j2_pt, j2_eta, j2_phi = pt_eta_phi(j2)

    dphi_as_labeled = kinematics.delta_phi(j1_phi, j2_phi)
    dphi_eta_ordered = kinematics.eta_ordered_dphi_jj(j1, j2)

    obs = plotting.get_observables(
        H_pt, H_eta, H_phi, j1_pt, j1_eta, j1_phi, j2_pt, j2_eta, dphi_as_labeled,
    )
    obs["dphi_eta_ordered"] = dphi_eta_ordered
    return obs


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
    truth_obs = build_observables(truth_fv)

    # ── reco four-vectors: decode reco encoding, use first two pT-ordered
    #    jet slots as a "detector-level j1/j2" proxy (jets are pT-ordered
    #    in the encoded reco array regardless of parton_ordering, which
    #    only affects truth) ──
    decoded_reco = kinematics.decode_domain(X_reco, resolved["reco"], "reco", max_jets=max_jets)
    H_reco_fv = kinematics.four_vector(
        decoded_reco["H_reco"]["pt"], decoded_reco["H_reco"]["eta"],
        decoded_reco["H_reco"]["phi"], mass=decoded_reco["H_reco"].get("mass", 0.0),
    )
    jet = decoded_reco["jet_reco"]
    j1_reco_fv = kinematics.four_vector(jet["pt"][:, 0], jet["eta"][:, 0], jet["phi"][:, 0],
                                          mass=jet.get("mass", np.zeros_like(jet["pt"]))[:, 0])
    j2_reco_fv = kinematics.four_vector(jet["pt"][:, 1], jet["eta"][:, 1], jet["phi"][:, 1],
                                          mass=jet.get("mass", np.zeros_like(jet["pt"]))[:, 1])
    reco_obs = build_observables({"H": H_reco_fv, "j1": j1_reco_fv, "j2": j2_reco_fv})

    # ── unfolded (flow) four-vectors: sample posterior, one draw/event
    #    for closure comparison (matches sample count of truth/reco) ──
    X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)
    samples_scaled = sample_posterior_batch(
        model, X_reco_scaled, truth_dim, n_samples_per_event=n_samples,
        device=device, batch_size=batch_size,
    )
    samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
    flow_fv = kinematics.reconstruct_event(samples, resolved["truth"])
    flow_obs = build_observables(flow_fv)

    return {"truth": truth_obs, "reco": reco_obs, "flow": flow_obs}


def run_evaluate(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    val_fold = torch.load(args.checkpoint, map_location=device)["val_fold"]
    print(f"evaluating on held-out fold {val_fold} (the checkpoint's own validation fold)")

    reco_dim = kinematics.total_dim(bundle["resolved"]["reco"], max_jets=bundle["max_jets"])
    truth_dim = kinematics.total_dim(bundle["resolved"]["truth"], max_jets=bundle["max_jets"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with pd.HDFStore(args.preprocessed, mode="r") as store:
        scenarios = [k.strip("/") for k in store.keys()]

    for scenario in scenarios:
        print(f"\n[{scenario}]")
        df_val = load_val_fold(args.preprocessed, scenario, val_fold, reco_dim, truth_dim)
        print(f"  {len(df_val)} held-out events")
        if len(df_val) == 0:
            print("  skipping -- no events in this fold for this scenario")
            continue

        result = evaluate_scenario(scenario, df_val, bundle, args.n_samples, device, args.batch_size)

        plot_specs = list(plotting.DEFAULT_PLOT_SPECS) + [
            ("dphi_eta_ordered", np.linspace(-np.pi, np.pi, 20),
             r"$\Delta\phi_{jj}$ (eta-ordered, CP)", r"$\Delta\phi_{jj}$ (CP convention)"),
        ]
        fig = plotting.plot_closure(
            result["truth"], result["reco"], result["flow"],
            plot_specs=plot_specs, title=f"Closure: {scenario} (fold {val_fold})",
        )
        out_path = Path(args.output_dir) / f"closure_{scenario}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
    return p


def main():
    args = build_arg_parser().parse_args()
    run_evaluate(args)


if __name__ == "__main__":
    main()