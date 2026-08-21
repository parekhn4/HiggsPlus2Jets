"""
Usage
    python scripts/sanity_check_truth_dphi_legacy.py --config configs/no_energy.yaml \
        --output truth_dphi_sanity_legacy.pdf

Same plot as scripts/sanity_check_truth_dphi.py (parton-level dphi_jj, all three
at/bt scenarios overlaid, both pT-ordered and eta-ordered conventions) but reading
the OLD Delphes_Data/ files (tree "Delphes", Particle.PID/Status truth extraction)
instead of Pratik's new fromPratik_sim_delphes/ files -- a cross-check for whether
whatever looked off in the new-data plot is also present in the previously-trusted
old data, or is specific to the new files/processing.
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
import inference.preprocessing_inference as inference_prep
from scripts.build_forMarcel_handoff import truth_four_vectors_from_extracted

SCENARIOS = ["at_1_bt_0", "at_0_bt_1", "at_1_bt_1"]


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data"]["input_dir"]
    scenario_files = inference_prep.discover_scenario_files(data_dir, config)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)
    colors = {"at_1_bt_0": "C0", "at_0_bt_1": "C1", "at_1_bt_1": "C2"}

    for scenario in SCENARIOS:
        path = scenario_files[scenario]
        print(f"[{scenario}] reading {path}")
        native = training.read_native_arrays(str(path), config)
        extracted = training.select_and_extract(native, config)
        print(f"  {extracted['n_events']} events pass selection")

        truth_fv = truth_four_vectors_from_extracted(extracted)
        obs = kinematics.build_observables(truth_fv)

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
    fig.suptitle("Parton-level dphi_jj sanity check (OLD Delphes_Data) -- all three at/bt scenarios")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", help="Override config's data.input_dir")
    p.add_argument("--output", required=True)
    p.add_argument("--n-bins", type=int, default=40)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
