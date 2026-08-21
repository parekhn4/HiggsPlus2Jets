"""
Usage
    python scripts/compare_old_new_truth_dphi.py --config configs/no_energy.yaml \
        --data-dir fromPratik_sim_delphes/ --mjj-cut 100.0 \
        --output ../pratik_eval/compare_old_new_truth_dphi.pdf --n-bins 40

Parton-level (truth) dphi_jj shape comparison, OLD (Delphes_Data) vs NEW
(fromPratik_sim_delphes) data, one row per scenario plus a pooled row, both
pT-ordered and eta-ordered conventions -- the direct check for whether the
old-checkpoint-on-new-data evaluation looks off because the underlying truth
phase space itself differs (see CLAUDE.md "New data source" -- new data uses a
much weaker hard-scatter m_jj cut than old). Each panel overlays old / new-uncut /
new with m_jj > --mjj-cut reinstated, and prints a real sum-of-squared-density-
difference (not a PDF read) for both "new vs old" comparisons, per scenario and
pooled.
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

OLD_COLOR = "0.4"
NEW_COLOR = "C1"
CUT_COLOR = "C2"


def hist_density(values, bins):
    counts, _ = np.histogram(values, bins=bins, density=True)
    return counts


def sse(a, b):
    return float(np.sum((a - b) ** 2))


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    old_data_dir = args.old_data_dir or config["data"]["input_dir"]
    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)

    per_scenario = {}
    for scenario in SCENARIOS:
        old_path = Path(old_data_dir) / scenario / "tag_1_delphes_events.root"
        native = training.read_native_arrays(str(old_path), config)
        extracted = training.select_and_extract(native, config)
        old_fv = truth_four_vectors_from_extracted(extracted)
        old_obs = kinematics.build_observables(old_fv)

        new_path = Path(args.data_dir) / scenario / "merged.root"
        print(f"[{scenario}] reading {new_path}")
        _, new_fv = read_and_select(str(new_path), config)
        new_obs = kinematics.build_observables(new_fv)
        mjj = invariant_mass(new_fv["j1"], new_fv["j2"])
        keep = mjj > args.mjj_cut
        cut_dphi = new_obs["dphi"][keep]
        cut_dphi_eta = new_obs["dphi_eta_ordered"][keep]

        print(f"  old: {len(old_obs['dphi'])} events; new: {len(new_obs['dphi'])} events "
              f"uncut, {keep.sum()} after m_jj > {args.mjj_cut} GeV ({100 * keep.mean():.1f}%)")

        per_scenario[scenario] = {
            "old_dphi": old_obs["dphi"], "old_dphi_eta": old_obs["dphi_eta_ordered"],
            "new_dphi": new_obs["dphi"], "new_dphi_eta": new_obs["dphi_eta_ordered"],
            "cut_dphi": cut_dphi, "cut_dphi_eta": cut_dphi_eta,
        }

    pooled = {
        key: np.concatenate([per_scenario[s][key] for s in SCENARIOS])
        for key in ["old_dphi", "old_dphi_eta", "new_dphi", "new_dphi_eta", "cut_dphi", "cut_dphi_eta"]
    }
    rows = SCENARIOS + ["pooled"]
    data = {**per_scenario, "pooled": pooled}

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 4.2 * len(rows)))
    print()
    for i, row_label in enumerate(rows):
        d = data[row_label]
        old_d = hist_density(d["old_dphi"], bins)
        new_d = hist_density(d["new_dphi"], bins)
        cut_d = hist_density(d["cut_dphi"], bins)
        old_de = hist_density(d["old_dphi_eta"], bins)
        new_de = hist_density(d["new_dphi_eta"], bins)
        cut_de = hist_density(d["cut_dphi_eta"], bins)

        print(f"[{row_label}] pT-ordered  SSE(new_uncut vs old)={sse(new_d, old_d):.5f}  "
              f"SSE(new_cut vs old)={sse(cut_d, old_d):.5f}")
        print(f"[{row_label}] eta-ordered SSE(new_uncut vs old)={sse(new_de, old_de):.5f}  "
              f"SSE(new_cut vs old)={sse(cut_de, old_de):.5f}")

        for j, (arr_old, arr_new, arr_cut, title) in enumerate([
            (d["old_dphi"], d["new_dphi"], d["cut_dphi"], "pT-ordered (training convention)"),
            (d["old_dphi_eta"], d["new_dphi_eta"], d["cut_dphi_eta"], r"$\eta$-ordered (CP convention)"),
        ]):
            ax = axes[i][j]
            ax.hist(arr_old, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color=OLD_COLOR, label="old")
            ax.hist(arr_new, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color=NEW_COLOR, label="new, uncut")
            ax.hist(arr_cut, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color=CUT_COLOR, label=f"new, $m_{{jj}}$>{args.mjj_cut:.0f} GeV")
            ax.set_title(f"{row_label} -- {title}", fontsize=10)
            ax.set_xlabel(r"$\Delta\phi_{jj}$")
            ax.set_ylabel("density")
            ax.set_xlim(-np.pi, np.pi)
            ax.legend(fontsize=7)

    fig.suptitle("Parton-level $\\Delta\\phi_{jj}$: old vs. new data, per scenario + pooled", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True, help="fromPratik_sim_delphes/ (new data)")
    p.add_argument("--old-data-dir", default=None,
                    help="Delphes_Data/ (old data). Defaults to the config's data.input_dir.")
    p.add_argument("--mjj-cut", type=float, default=100.0)
    p.add_argument("--output", required=True)
    p.add_argument("--n-bins", type=int, default=40)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
