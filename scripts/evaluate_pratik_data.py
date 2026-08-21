"""
Usage
    python scripts/evaluate_pratik_data.py --checkpoint runs/2026-07-10_no_energy_16b/model.pt \
        --config configs/no_energy.yaml --data-dir ../fromPratik_sim_delphes/ \
        --output-dir ../pratik_eval/ --n-bins 40

One-off analysis script (not the durable pipeline -- see CLAUDE.md "New data source:
fromPratik_sim_delphes/") evaluating an already-trained checkpoint on Pratik's larger-
statistics samples (~2.7M events total, "Events" tree, different branch names/structure
than the old Delphes-native files). Produces per-scenario + pooled dphi_jj closure plots,
finer-binned than the default 19 bins, in both the training (pT-ordered) and CP
(eta-ordered) conventions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import uproot
import awkward as ak
import yaml

import core.kinematics as kinematics
import plotting.plotting as plotting
import inference.preprocessing_inference as inference_prep
from inference.inference import load_checkpoint_bundle, sample_posterior_batch

SCENARIOS = ["at_0_bt_1", "at_1_bt_0", "at_1_bt_1"]


def first_subfile_entry_stop(path: str) -> int:
    """
    merged.root is 25 hadd'd pre-merge files concatenated; eventNumber resets
    (goes non-monotonic) at each boundary. Returns the row count of just the
    FIRST pre-merge file, so callers can restrict to un-merged data only --
    e.g. to check whether an anomaly also appears within a single original
    file, or only after merging (see CLAUDE.md "New data source" section).
    """
    with uproot.open(path) as f:
        en = f["Events"].arrays(["eventNumber"], library="np")["eventNumber"]
    resets = np.where(np.diff(en) < 0)[0]
    return int(resets[0] + 1) if len(resets) else len(en)


def read_and_select(path: str, config: dict, entry_stop: int | None = None):
    """
    Read one Pratik-format scenario file, apply the standard object-quality +
    event-selection + mass-window cuts (reusing the existing, tested reco
    pipeline via a native-name rename), and build truth H/j1/j2 (pT-ordered)
    directly from the already-pre-filtered hsjet/higgs branches. Returns
    (reco_fv, truth_fv, X_reco[encoded, unscaled], resolved_reco-shaped extracted dict).

    entry_stop: if given, only reads the first N raw rows (before selection)
    -- e.g. pass first_subfile_entry_stop(path) to restrict to just the first
    of the 25 pre-merge files.
    """
    with uproot.open(path) as f:
        raw = f["Events"].arrays(
            ["jet_pt", "jet_eta", "jet_phi", "jet_m",
             "phot_pt", "phot_eta", "phot_phi", "phot_e",
             "hsjet_pid", "hsjet_pt", "hsjet_eta", "hsjet_phi", "hsjet_m",
             "higgs_pt", "higgs_eta", "higgs_phi", "higgs_m"],
            library="ak", entry_stop=entry_stop,
        )

    # rename into the shape inference/preprocessing_inference.py already expects,
    # so its (tested) object-quality-cut + selection + reco-extraction logic can
    # be reused completely unchanged
    native = {
        "jet_pt": raw["jet_pt"], "jet_eta": raw["jet_eta"],
        "jet_phi": raw["jet_phi"], "jet_mass": raw["jet_m"],
        "photon_pt": raw["phot_pt"], "photon_eta": raw["phot_eta"],
        "photon_phi": raw["phot_phi"], "photon_E": raw["phot_e"],
    }
    native = inference_prep.apply_object_quality_cuts(native, config)
    mask = inference_prep.reco_selection_mask(native, config)
    extracted = inference_prep.extract_reco_quantities(native, mask, config)
    mass_ok = extracted["mass_ok"]

    # truth: hsjet is always exactly 2/event, higgs is flat 1/event -- both
    # already pre-filtered upstream (see CLAUDE.md), so this is just a
    # rename + pT-ordering swap, no PID/status masking needed
    p_pt = ak.to_numpy(raw["hsjet_pt"][mask])[mass_ok]
    p_eta = ak.to_numpy(raw["hsjet_eta"][mask])[mass_ok]
    p_phi = ak.to_numpy(raw["hsjet_phi"][mask])[mass_ok]
    p_mass = ak.to_numpy(raw["hsjet_m"][mask])[mass_ok]
    swap = kinematics.compute_swap_mask(p_pt, p_eta, "pt")
    for a in (p_pt, p_eta, p_phi, p_mass):
        tmp = a[swap, 0].copy()
        a[swap, 0] = a[swap, 1]
        a[swap, 1] = tmp

    h_pt = ak.to_numpy(raw["higgs_pt"][mask])[mass_ok]
    h_eta = ak.to_numpy(raw["higgs_eta"][mask])[mass_ok]
    h_phi = ak.to_numpy(raw["higgs_phi"][mask])[mass_ok]
    h_mass = ak.to_numpy(raw["higgs_m"][mask])[mass_ok]

    truth_fv = {
        "H": kinematics.four_vector(h_pt, h_eta, h_phi, mass=h_mass),
        "j1": kinematics.four_vector(p_pt[:, 0], p_eta[:, 0], p_phi[:, 0], mass=p_mass[:, 0]),
        "j2": kinematics.four_vector(p_pt[:, 1], p_eta[:, 1], p_phi[:, 1], mass=p_mass[:, 1]),
    }
    return extracted, truth_fv


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    model, resolved, max_jets, scaler = bundle["model"], bundle["resolved"], bundle["max_jets"], bundle["scaler"]
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)
    print(f"checkpoint: epoch {bundle['epoch']}, val_loss {bundle['val_loss']:.4f}, "
          f"parton_ordering: {resolved['parton_ordering']}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pooled = {"truth": [], "reco": [], "unfolded": []}

    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        print(f"\n[{scenario}] reading {path}")
        extracted, truth_fv = read_and_select(str(path), config)
        n_events = extracted["n_events"]
        print(f"  {n_events} events pass selection")

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
        write_plots(result, args.output_dir, scenario, args.n_bins)

        for key in pooled:
            pooled[key].append(result[key])

    print(f"\n[pooled, all {len(SCENARIOS)} scenarios]")
    pooled_obs = {
        domain: {key: np.concatenate([d[key] for d in obs_dicts]) for key in obs_dicts[0].keys()}
        for domain, obs_dicts in pooled.items()
    }
    write_plots(pooled_obs, args.output_dir, "pooled", args.n_bins)
    print(f"\nDone. Plots under {args.output_dir}/")


def write_plots(result: dict, output_dir: str, label: str, n_bins: int) -> None:
    plot_specs = [
        ("dphi", np.linspace(-np.pi, np.pi, n_bins + 1),
         r"$\Delta\phi_{jj}$ (pT-ordered, training convention)", r"$\Delta\phi_{jj}$"),
        ("dphi_eta_ordered", np.linspace(-np.pi, np.pi, n_bins + 1),
         r"$\Delta\phi_{jj}$ (eta-ordered, CP convention)", r"$\Delta\phi_{jj}$ (CP)"),
    ]
    fig = plotting.plot_closure(
        result["truth"], result["reco"], result["unfolded"],
        plot_specs=plot_specs, title=f"dphi_jj closure (single-draw): {label}", ncols=2,
    )
    out_path = Path(output_dir) / f"dphi_jj_closure_{label}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="Selection thresholds + used as the base config "
                                                     "(truth/reco variable resolution comes from the "
                                                     "checkpoint itself, not this file)")
    p.add_argument("--data-dir", required=True, help="Directory containing <scenario>/merged.root")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-bins", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
