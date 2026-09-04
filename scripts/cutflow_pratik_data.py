"""
Usage
    python scripts/cutflow_pratik_data.py --config configs/no_energy.yaml \
        --data-dir fromPratik_sim_delphes/ --mjj-cut 100.0

Sequential event-count cutflow for the new (Pratik) data, per scenario and pooled:
raw -> after object-quality + >=2 jet/photon event selection -> after the reco mass
window -> after the hard-scatter m_jj cut (the diagnostic cut from
scripts/check_mjj_cut_effect.py, not part of the standard pipeline selection).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import awkward as ak
import uproot
import yaml

import inference.preprocessing_inference as inference_prep
import core.kinematics as kinematics
from scripts.evaluate_pratik_data import SCENARIOS
from scripts.check_mjj_cut_effect import invariant_mass


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    totals = {"raw": 0, "after_quality_cuts": 0, "after_mass_window": 0, "after_mjj_cut": 0}

    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        with uproot.open(str(path)) as f:
            raw = f["Events"].arrays(
                ["jet_pt", "jet_eta", "jet_phi", "jet_m",
                 "phot_pt", "phot_eta", "phot_phi", "phot_e",
                 "hsjet_pt", "hsjet_eta", "hsjet_phi", "hsjet_m"],
                library="ak",
            )
        n_raw = len(raw["jet_pt"])

        native = {
            "jet_pt": raw["jet_pt"], "jet_eta": raw["jet_eta"],
            "jet_phi": raw["jet_phi"], "jet_mass": raw["jet_m"],
            "photon_pt": raw["phot_pt"], "photon_eta": raw["phot_eta"],
            "photon_phi": raw["phot_phi"], "photon_E": raw["phot_e"],
        }
        native = inference_prep.apply_object_quality_cuts(native, config)
        mask = inference_prep.reco_selection_mask(native, config)
        n_after_quality = int(ak.sum(mask))

        extracted = inference_prep.extract_reco_quantities(native, mask, config)
        n_after_mass = extracted["n_events"]

        p_pt = raw["hsjet_pt"][mask][extracted["mass_ok"]]
        p_eta = raw["hsjet_eta"][mask][extracted["mass_ok"]]
        p_phi = raw["hsjet_phi"][mask][extracted["mass_ok"]]
        p_mass = raw["hsjet_m"][mask][extracted["mass_ok"]]
        j1_fv = kinematics.four_vector(ak.to_numpy(p_pt[:, 0]), ak.to_numpy(p_eta[:, 0]),
                                        ak.to_numpy(p_phi[:, 0]), mass=ak.to_numpy(p_mass[:, 0]))
        j2_fv = kinematics.four_vector(ak.to_numpy(p_pt[:, 1]), ak.to_numpy(p_eta[:, 1]),
                                        ak.to_numpy(p_phi[:, 1]), mass=ak.to_numpy(p_mass[:, 1]))
        mjj = invariant_mass(j1_fv, j2_fv)
        n_after_mjj = int((mjj > args.mjj_cut).sum())

        print(f"[{scenario}] raw={n_raw}  after_quality_cuts(>=2 jets,>=2 photons)={n_after_quality}  "
              f"after_mass_window={n_after_mass}  after_mjj>{args.mjj_cut:.0f}={n_after_mjj}")

        totals["raw"] += n_raw
        totals["after_quality_cuts"] += n_after_quality
        totals["after_mass_window"] += n_after_mass
        totals["after_mjj_cut"] += n_after_mjj

    print(f"\n[pooled] raw={totals['raw']}  after_quality_cuts={totals['after_quality_cuts']}  "
          f"after_mass_window={totals['after_mass_window']}  after_mjj>{args.mjj_cut:.0f}={totals['after_mjj_cut']}")
    print(f"\n[pooled fractions of raw] after_quality_cuts={totals['after_quality_cuts']/totals['raw']:.4f}  "
          f"after_mass_window={totals['after_mass_window']/totals['raw']:.4f}  "
          f"after_mjj_cut={totals['after_mjj_cut']/totals['raw']:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mjj-cut", type=float, default=100.0)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
