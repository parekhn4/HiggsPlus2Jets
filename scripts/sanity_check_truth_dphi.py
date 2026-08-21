"""
Usage
    python scripts/sanity_check_truth_dphi.py --config configs/no_energy.yaml \
        --data-dir ../fromPratik_sim_delphes/ --output truth_dphi_sanity.pdf

Quick sanity check: parton-level (truth) dphi_jj only, all three at/bt scenarios
overlaid on the same axes, both the pT-ordered (training) and eta-ordered (CP)
conventions -- no reco, no unfolding, no model needed. Reuses
scripts/evaluate_pratik_data.py's read_and_select for the actual data reading.
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
from scripts.evaluate_pratik_data import read_and_select, first_subfile_entry_stop, SCENARIOS


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)
    colors = {"at_1_bt_0": "C0", "at_0_bt_1": "C1", "at_1_bt_1": "C2"}

    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        entry_stop = None
        if args.first_subfile_only:
            entry_stop = first_subfile_entry_stop(str(path))
            print(f"[{scenario}] first pre-merge file only: {entry_stop} raw rows")
        print(f"[{scenario}] reading {path}")
        extracted, truth_fv = read_and_select(str(path), config, entry_stop=entry_stop)
        obs = kinematics.build_observables(truth_fv)
        print(f"  {len(obs['dphi'])} events")

        axes[0].hist(obs["dphi"], bins=bins, density=True, histtype="step",
                     linewidth=1.8, label=scenario, color=colors[scenario])
        axes[1].hist(obs["dphi_eta_ordered"], bins=bins, density=True, histtype="step",
                     linewidth=1.8, label=scenario, color=colors[scenario])

    axes[0].set_title(r"Truth $\Delta\phi_{jj}$ -- pT-ordered (training convention)")
    axes[1].set_title(r"Truth $\Delta\phi_{jj}$ -- $\eta$-ordered (CP convention)")
    for ax in axes:
        ax.set_xlabel(r"$\Delta\phi_{jj}$")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_xlim(-np.pi, np.pi)
    subtitle = " (first pre-merge file only, unmerged)" if args.first_subfile_only else " (full merged.root)"
    fig.suptitle(f"Parton-level dphi_jj sanity check{subtitle} -- all three at/bt scenarios")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-bins", type=int, default=40)
    p.add_argument("--first-subfile-only", action="store_true",
                    help="Restrict to just the first of the 25 pre-merge files per scenario "
                         "(detected via eventNumber reset), to check whether an anomaly seen "
                         "in the full merged.root is present in unmerged data too.")
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
