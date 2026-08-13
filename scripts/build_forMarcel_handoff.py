"""
Usage
    python scripts/build_forMarcel_handoff.py --config configs/no_energy.yaml --output-dir ../forMarcel/

Replaces the old scripts/build_reco_handoff.py -- now builds the reco AND the
selection-applied truth four-vector handoffs together, from a single read/
selection pass per scenario (both come out of training.preprocessing_training.
select_and_extract already), instead of two separate passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

import core.kinematics as kinematics
import training.preprocessing_training as training
import inference.preprocessing_inference as inference_prep


def four_vectors_to_dataframe(fv: dict, event_id: np.ndarray) -> pd.DataFrame:
    """
    {"H": (N,4), "j1": (N,4), "j2": (N,4)} of (E, px, py, pz) -> a flat
    13-column dataframe: a leading AUX_event_id (0-based index into the
    source ROOT file) plus the same 12 cartesian physics columns
    forMarcel/'s original (no-cuts) truth handoff uses. AUX_event_id is
    required here, unlike that original file -- both outputs of this script
    have selection cuts applied, so row i is not event i in the source file.
    """
    cols = {"AUX_event_id": event_id}
    for name in ("H", "j1", "j2"):
        v = fv[name]
        cols[f"{name}_E"] = v[:, 0]
        cols[f"{name}_px"] = v[:, 1]
        cols[f"{name}_py"] = v[:, 2]
        cols[f"{name}_pz"] = v[:, 3]
    df = pd.DataFrame(cols)
    for c in df.columns:
        if c != "AUX_event_id":
            df[c] = df[c].astype(np.float32)
    return df


def reco_four_vectors_from_extracted(extracted: dict) -> dict:
    """
    {"H","j1","j2"} four-vectors straight from select_and_extract's raw
    physical reco quantities -- j1/j2 are the two hardest (highest-pT) reco
    jets, Delphes' native ordering (same convention as
    kinematics.reco_four_vectors), but built directly rather than round-
    tripping through a specific config's encode/decode -- doesn't depend on
    which reco variables a given config happens to select for training.
    """
    H = extracted["H_reco"]
    jet = extracted["jet_reco"]
    return {
        "H": kinematics.four_vector(H["pt"], H["eta"], H["phi"], mass=H["mass"]),
        "j1": kinematics.four_vector(jet["pt"][:, 0], jet["eta"][:, 0], jet["phi"][:, 0],
                                       mass=jet["mass"][:, 0]),
        "j2": kinematics.four_vector(jet["pt"][:, 1], jet["eta"][:, 1], jet["phi"][:, 1],
                                       mass=jet["mass"][:, 1]),
    }


def truth_four_vectors_from_extracted(extracted: dict) -> dict:
    """
    {"H","j1","j2"} truth four-vectors from select_and_extract's raw
    physical quantities, with the *same* selection already applied as the
    reco side (>=2 quality photons/jets, pT/eta acceptance, mass window,
    exactly 1 truth Higgs + 2 hard partons) -- unlike forMarcel/'s original
    truth handoff, which has no cuts at all. j2's phi isn't stored directly
    (matches the training-side truth schema) -- recovered from j1's phi and
    the event-level dphi_jj, same as kinematics.recover_phi_j2 everywhere
    else in this project.
    """
    H = extracted["H_truth"]
    j1 = extracted["j1_truth"]
    j2 = extracted["j2_truth"]
    j2_phi = kinematics.recover_phi_j2(j1["phi"], extracted["event_truth"]["dphi_jj"])
    return {
        "H": kinematics.four_vector(H["pt"], H["eta"], H["phi"], mass=H["mass"]),
        "j1": kinematics.four_vector(j1["pt"], j1["eta"], j1["phi"], mass=j1["mass"]),
        "j2": kinematics.four_vector(j2["pt"], j2["eta"], j2_phi, mass=j2["mass"]),
    }


def run_build_handoff(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data"]["input_dir"]
    scenario_files = inference_prep.discover_scenario_files(data_dir, config)
    if not scenario_files:
        print(f"No scenario ROOT files found under {data_dir}.", file=sys.stderr)
        sys.exit(1)

    reco_dir = Path(args.output_dir) / "reco"
    truth_dir = Path(args.output_dir) / "truth_selected"
    reco_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name, path in scenario_files.items():
        print(f"[{scenario_name}] reading {path}")
        native = training.read_native_arrays(str(path), config)
        extracted = training.select_and_extract(native, config)
        n_events = extracted["n_events"]
        print(f"  {n_events} events pass selection (>=2 quality photons, >=2 quality jets, "
              f"pT/eta acceptance, mass window, exactly 1 truth Higgs + 2 hard partons)")
        event_id = extracted["event_id"]

        reco_fv = reco_four_vectors_from_extracted(extracted)
        reco_df = four_vectors_to_dataframe(reco_fv, event_id)
        if not np.isfinite(reco_df.drop(columns=["AUX_event_id"]).to_numpy()).all():
            raise ValueError(f"[{scenario_name}] non-finite values in reco four-vectors")
        reco_path = reco_dir / f"{scenario_name}_reco_four_vectors.h5"
        reco_df.to_hdf(reco_path, key="events", mode="w")
        h_mass_reco = np.sqrt(np.maximum(reco_df["H_E"] ** 2 - reco_df["H_px"] ** 2
                                          - reco_df["H_py"] ** 2 - reco_df["H_pz"] ** 2, 0))
        print(f"  [reco]  wrote {len(reco_df)} events -> {reco_path}  "
              f"(H mass {h_mass_reco.mean():.2f} +/- {h_mass_reco.std():.2f} GeV)")

        truth_fv = truth_four_vectors_from_extracted(extracted)
        truth_df = four_vectors_to_dataframe(truth_fv, event_id)
        if not np.isfinite(truth_df.drop(columns=["AUX_event_id"]).to_numpy()).all():
            raise ValueError(f"[{scenario_name}] non-finite values in truth four-vectors")
        truth_path = truth_dir / f"{scenario_name}_truth_four_vectors_selected.h5"
        truth_df.to_hdf(truth_path, key="events", mode="w")
        h_mass_truth = np.sqrt(np.maximum(truth_df["H_E"] ** 2 - truth_df["H_px"] ** 2
                                           - truth_df["H_py"] ** 2 - truth_df["H_pz"] ** 2, 0))
        print(f"  [truth] wrote {len(truth_df)} events -> {truth_path}  "
              f"(H mass {h_mass_truth.mean():.2f} +/- {h_mass_truth.std():.2f} GeV)")

        # both files come from the exact same selection pass, so their
        # event sets must line up 1:1 in the same row order
        assert (reco_df["AUX_event_id"].to_numpy() == truth_df["AUX_event_id"].to_numpy()).all(), \
            f"[{scenario_name}] reco/truth AUX_event_id mismatch -- should be impossible"

    print(f"\nDone. Files under {args.output_dir}/reco/ and {args.output_dir}/truth_selected/")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build both the reco-level (detector) and selection-applied "
                     "truth-level four-vector handoffs -- H (from diphotons/truth Higgs) "
                     "+ the two hardest (highest-pT, or pT-ordered for truth per "
                     "parton_ordering) j1/j2, cartesian four-vectors, same physics "
                     "columns as forMarcel/'s original (no-cuts) truth handoff plus a "
                     "leading AUX_event_id -- for every event passing the standard "
                     "selection (which, unlike the original truth handoff, both outputs "
                     "here actually have applied)."
    )
    p.add_argument("--config", required=True, help="Model config YAML (selection + parton_ordering)")
    p.add_argument("--data-dir", help="Override config's data.input_dir")
    p.add_argument("--output-dir", required=True,
                    help="Parent directory -- writes <dir>/reco/<scenario>_reco_four_vectors.h5 "
                         "and <dir>/truth_selected/<scenario>_truth_four_vectors_selected.h5")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_build_handoff(args)


if __name__ == "__main__":
    main()
