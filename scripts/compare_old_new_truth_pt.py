"""
Usage
    python scripts/compare_old_new_truth_pt.py --config configs/no_energy.yaml \
        --data-dir fromPratik_sim_delphes/ --mjj-cut 100.0 \
        --output ../pratik_eval/compare_old_new_truth_pt.pdf --n-bins 31

Follow-up to scripts/compare_old_new_truth_dphi.py and
scripts/old_data_stats_visibility_check.py: the j2_pt wobble seen in the new-data closure
plots was ruled out as a within-old-data statistics artifact (unchanged whether the old
val fold is unfolded once or 36x). This checks the other live hypothesis directly -- does
j2_pt's TRUTH-level marginal distribution itself still differ between old and new data even
after the m_jj>100 cut (which fixes dphi_jj's bulk shape), i.e. is the m_jj cut a coarse
post-hoc proxy that doesn't fully harmonize the phase space? Also checks j1_pt and H_pt as
controls -- H_pt in particular should be close to untouched by a dijet-system cut, so a
clean H_pt match alongside a real j1_pt/j2_pt mismatch would isolate the effect to the
dijet kinematics specifically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import yaml

import core.kinematics as kinematics
import training.preprocessing_training as training
from scripts.build_forMarcel_handoff import truth_four_vectors_from_extracted
from scripts.evaluate_pratik_data import read_and_select, SCENARIOS
from scripts.check_mjj_cut_effect import invariant_mass

OBS_SPECS = [
    ("j2_pt", np.linspace(0, 300, 31), r"$p_{T,j2}$ [GeV]"),
    ("j1_pt", np.linspace(0, 300, 31), r"$p_{T,j1}$ [GeV]"),
    ("H_pt", np.linspace(0, 400, 41), r"$p_{T,H}$ [GeV]"),
]


def sse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    old_data_dir = args.old_data_dir or config["data"]["input_dir"]

    old_per_scenario = {}
    for scenario in SCENARIOS:
        path = f"{old_data_dir}/{scenario}/tag_1_delphes_events.root"
        native = training.read_native_arrays(path, config)
        extracted = training.select_and_extract(native, config)
        truth_fv = truth_four_vectors_from_extracted(extracted)
        obs = kinematics.build_observables(truth_fv)
        old_per_scenario[scenario] = obs
        print(f"[old, {scenario}] {len(obs['j2_pt'])} events")

    new_per_scenario = {}
    for scenario in SCENARIOS:
        path = f"{args.data_dir}/{scenario}/merged.root"
        print(f"[{scenario}] reading {path}")
        _, truth_fv = read_and_select(path, config)
        obs = kinematics.build_observables(truth_fv)
        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        new_per_scenario[scenario] = {k: v[keep] for k, v in obs.items()}
        print(f"  {len(obs['j2_pt'])} events, {keep.sum()} after m_jj > {args.mjj_cut} GeV")

    old_pooled = {key: np.concatenate([old_per_scenario[s][key] for s in SCENARIOS]) for key, _, _ in OBS_SPECS}
    new_pooled = {key: np.concatenate([new_per_scenario[s][key] for s in SCENARIOS]) for key, _, _ in OBS_SPECS}

    rows = SCENARIOS + ["pooled"]
    old_data = {**old_per_scenario, "pooled": old_pooled}
    new_data = {**new_per_scenario, "pooled": new_pooled}

    fig, axes = plt.subplots(len(rows), len(OBS_SPECS), figsize=(6 * len(OBS_SPECS), 4.2 * len(rows)))
    print()
    for i, row_label in enumerate(rows):
        for j, (key, bins, xlabel) in enumerate(OBS_SPECS):
            old_h, _ = np.histogram(old_data[row_label][key], bins=bins, density=True)
            new_h, _ = np.histogram(new_data[row_label][key], bins=bins, density=True)
            s = sse(new_h, old_h)
            print(f"[{row_label}] {key}: SSE(new_cut vs old) = {s:.6f}")

            ax = axes[i][j]
            ax.hist(old_data[row_label][key], bins=bins, density=True, histtype="step",
                    linewidth=1.8, color="0.4", label="old, truth")
            ax.hist(new_data[row_label][key], bins=bins, density=True, histtype="step",
                    linewidth=1.8, color="C1", label=f"new, truth, $m_{{jj}}$>{args.mjj_cut:.0f} GeV")
            ax.set_title(f"{row_label} -- {xlabel}", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.legend(fontsize=7)

    fig.suptitle("Truth-level pT marginals: old vs. new ($m_{jj}$-cut) data", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--old-data-dir", default=None)
    p.add_argument("--mjj-cut", type=float, default=100.0)
    p.add_argument("--n-bins", type=int, default=31)
    p.add_argument("--output", required=True)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
