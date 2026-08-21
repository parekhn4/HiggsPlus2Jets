"""
Usage
    python scripts/closure_pratik_mjj_cut.py \
        --checkpoint runs/2026-08-11_no_energy_16b_energy_score/model.pt \
        --config configs/no_energy.yaml --data-dir fromPratik_sim_delphes/ \
        --mjj-cut 100.0 --output-dir ../pratik_eval/closure_mjj_cut/ --bin-multiplier 4

Single-draw closure plots (all standard observables: dphi, dphi_eta_ordered, deta,
j1/j2 pt+eta, H pt+eta+phi), scenario-specific + pooled, on the ENTIRE Pratik merged.root
files (no train/val/test split needed here -- this checkpoint was trained on the OLD
Delphes_Data/, so none of the new data is contamination) with the hard-scatter m_jj > 100 GeV
cut reinstated first (see scripts/check_mjj_cut_effect.py / CLAUDE.md -- this cut brings the
new data's dphi_jj shape back in line with the old data; only real physics processes should
be evaluated under matched-cut phase space). Bins default to 4x DEFAULT_PLOT_SPECS' bin
counts, since ~640k events/scenario after the cut comfortably supports finer binning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

import core.kinematics as kinematics
import plotting.plotting as plotting
import inference.preprocessing_inference as inference_prep
from inference.inference import load_checkpoint_bundle, sample_posterior_batch
from scripts.evaluate_pratik_data import read_and_select, SCENARIOS
from scripts.check_mjj_cut_effect import invariant_mass


def subset_extracted(extracted: dict, keep: np.ndarray) -> dict:
    return {
        "n_events": int(keep.sum()),
        "mass_ok": extracted["mass_ok"],
        "event_id": extracted["event_id"][keep],
        "H_reco": {k: v[keep] for k, v in extracted["H_reco"].items()},
        "jet_reco": {k: v[keep] for k, v in extracted["jet_reco"].items()},
        "event_reco": {k: v[keep] for k, v in extracted["event_reco"].items()},
    }


def finer_plot_specs(multiplier: int) -> list:
    specs = []
    for key, bins, xlabel, subtitle in plotting.DEFAULT_PLOT_SPECS:
        n_bins = (len(bins) - 1) * multiplier
        specs.append((key, np.linspace(bins[0], bins[-1], n_bins + 1), xlabel, subtitle))
    specs.append(("dphi_eta_ordered", np.linspace(-np.pi, np.pi, 20 * multiplier),
                   r"$\Delta\phi_{jj}$ (CP)", r"$\Delta\phi_{jj}$ ($\eta$-ordered, CP convention)"))
    return specs


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    model, resolved, max_jets, scaler = bundle["model"], bundle["resolved"], bundle["max_jets"], bundle["scaler"]
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)
    # load_checkpoint_bundle() doesn't surface residual_penalty_type -- read it directly
    # off the raw checkpoint dict instead, just for this identifying print
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: epoch {bundle['epoch']}, val_loss {bundle['val_loss']:.4f}, "
          f"parton_ordering: {resolved['parton_ordering']}, "
          f"residual_penalty_type: {raw_ckpt.get('residual_penalty_type')}, "
          f"residual_penalty_weight: {raw_ckpt.get('residual_penalty_weight')}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_specs = finer_plot_specs(args.bin_multiplier)

    pooled = {"truth": [], "reco": [], "unfolded": []}

    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        print(f"\n[{scenario}] reading {path}")
        extracted, truth_fv = read_and_select(str(path), config)
        n_events = extracted["n_events"]

        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        print(f"  {n_events} events pass selection, {keep.sum()} pass m_jj > "
              f"{args.mjj_cut} GeV ({100 * keep.mean():.1f}%)")

        extracted = subset_extracted(extracted, keep)
        truth_fv = {k: v[keep] for k, v in truth_fv.items()}

        X_reco = kinematics.encode_domain(extracted, resolved["reco"], "reco", max_jets=max_jets)
        reco_fv = kinematics.reco_four_vectors(X_reco, resolved["reco"], max_jets)

        X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)
        torch.manual_seed(args.seed)
        samples_scaled = sample_posterior_batch(
            model, X_reco_scaled, truth_dim, n_samples_per_event=1,
            device=device, batch_size=args.batch_size,
        )
        samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)
        unfolded_fv = kinematics.reconstruct_event(samples, resolved["truth"])

        result = {
            "truth": kinematics.build_observables(truth_fv),
            "reco": kinematics.build_observables(reco_fv),
            "unfolded": kinematics.build_observables(unfolded_fv),
        }
        write_plots(result, args.output_dir, scenario, plot_specs)

        for key in pooled:
            pooled[key].append(result[key])

    print(f"\n[pooled, all {len(SCENARIOS)} scenarios]")
    pooled_obs = {
        domain: {key: np.concatenate([d[key] for d in obs_dicts]) for key in obs_dicts[0].keys()}
        for domain, obs_dicts in pooled.items()
    }
    write_plots(pooled_obs, args.output_dir, "pooled", plot_specs)
    print(f"\nDone. Plots under {args.output_dir}/")


def write_plots(result: dict, output_dir: str, label: str, plot_specs: list) -> None:
    fig = plotting.plot_closure(
        result["truth"], result["reco"], result["unfolded"],
        plot_specs=plot_specs, ncols=3,
        title=f"Closure (single-draw), $m_{{jj}}$>100 GeV cut: {label}",
    )
    out_path = Path(output_dir) / f"closure_{label}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mjj-cut", type=float, default=100.0)
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
