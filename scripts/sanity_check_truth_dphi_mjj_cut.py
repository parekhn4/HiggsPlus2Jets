"""
Usage
    python scripts/sanity_check_truth_dphi_mjj_cut.py --config configs/no_energy.yaml \
        --data-dir fromPratik_sim_delphes/ --mjj-cut 100.0 \
        --output ../pratik_eval/truth_dphi_sanity_mjj_cut.pdf --n-bins 400

Same per-scenario overlay as scripts/sanity_check_truth_dphi.py (parton-level dphi_jj,
all three at/bt scenarios on the same axes, both pT-ordered and eta-ordered conventions),
but with the hard-scatter (truth parton) m_jj > --mjj-cut cut reinstated first -- see
scripts/check_mjj_cut_effect.py and CLAUDE.md "New data source" section for why: the new
samples were generated with a much weaker m_jj cut (mmjj=20) than whatever the old samples
used, and reinstating a stronger one (mmjj=100, per the advisor) brings the shape back in
line with the old data.
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
from scripts.evaluate_pratik_data import read_and_select, SCENARIOS
from scripts.check_mjj_cut_effect import invariant_mass


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)
    colors = {"at_1_bt_0": "C0", "at_0_bt_1": "C1", "at_1_bt_1": "C2"}

    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        print(f"[{scenario}] reading {path}")
        extracted, truth_fv = read_and_select(str(path), config)
        obs = kinematics.build_observables(truth_fv)
        n_events = len(obs["dphi"])

        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        print(f"  {n_events} events, {keep.sum()} pass m_jj > {args.mjj_cut} GeV "
              f"({100 * keep.mean():.1f}%)")

        axes[0].hist(obs["dphi"][keep], bins=bins, density=True, histtype="step",
                     linewidth=1.8, label=scenario, color=colors[scenario])
        axes[1].hist(obs["dphi_eta_ordered"][keep], bins=bins, density=True, histtype="step",
                     linewidth=1.8, label=scenario, color=colors[scenario])

    axes[0].set_title(r"Truth $\Delta\phi_{jj}$ -- pT-ordered (training convention)")
    axes[1].set_title(r"Truth $\Delta\phi_{jj}$ -- $\eta$-ordered (CP convention)")
    for ax in axes:
        ax.set_xlabel(r"$\Delta\phi_{jj}$")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_xlim(-np.pi, np.pi)
    fig.suptitle(rf"Parton-level dphi_jj sanity check (new data, hard-scatter $m_{{jj}}$ > "
                 rf"{args.mjj_cut:.0f} GeV) -- all three at/bt scenarios")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mjj-cut", type=float, default=100.0)
    p.add_argument("--output", required=True)
    p.add_argument("--n-bins", type=int, default=400)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
