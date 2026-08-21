"""
Usage
    python scripts/check_mjj_cut_effect.py --config configs/no_energy.yaml \
        --data-dir fromPratik_sim_delphes/ --mjj-cut 100.0 \
        --output ../pratik_eval/mjj_cut_effect.pdf --n-bins 400

One-off follow-up to the "New data source" dphi_jj shape anomaly (see CLAUDE.md).
Advisor's hypothesis: the new samples were generated with a much weaker hard-scattering
m_jj cut (`set mmjj 20.0`) than whatever the old samples used (advisor: "mjj>100 cut is
chosen to make the topology more VBF-like"). Reinstates an m_jj > --mjj-cut cut on the
truth-level (hard-scatter parton) dijet system in the NEW data only, and overlays
uncut-new vs. mjj-cut-new vs. old, both dphi_jj conventions, to check whether the cut
alone reproduces the old shape.
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

OLD_COLOR = "0.4"
NEW_COLOR = "C1"
CUT_COLOR = "C2"


def invariant_mass(fv_a: np.ndarray, fv_b: np.ndarray) -> np.ndarray:
    fv_sum = fv_a + fv_b
    e, px, py, pz = fv_sum[:, 0], fv_sum[:, 1], fv_sum[:, 2], fv_sum[:, 3]
    m2 = e ** 2 - px ** 2 - py ** 2 - pz ** 2
    return np.sqrt(np.clip(m2, 0, None))


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    old_data_dir = args.old_data_dir or config["data"]["input_dir"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)

    # old data, pooled across scenarios, for reference shape
    old_dphi, old_dphi_eta = [], []
    for scenario in SCENARIOS:
        path = Path(old_data_dir) / scenario / "tag_1_delphes_events.root"
        native = training.read_native_arrays(str(path), config)
        extracted = training.select_and_extract(native, config)
        truth_fv = truth_four_vectors_from_extracted(extracted)
        obs = kinematics.build_observables(truth_fv)
        old_dphi.append(obs["dphi"])
        old_dphi_eta.append(obs["dphi_eta_ordered"])
    old_dphi = np.concatenate(old_dphi)
    old_dphi_eta = np.concatenate(old_dphi_eta)
    print(f"[old, pooled] {len(old_dphi)} events")

    # new data, pooled, uncut vs. mjj-cut
    new_dphi, new_dphi_eta = [], []
    cut_dphi, cut_dphi_eta = [], []
    for scenario in SCENARIOS:
        path = Path(args.data_dir) / scenario / "merged.root"
        print(f"[{scenario}] reading {path}")
        extracted, truth_fv = read_and_select(str(path), config)
        obs = kinematics.build_observables(truth_fv)
        n_events = len(obs["dphi"])
        new_dphi.append(obs["dphi"])
        new_dphi_eta.append(obs["dphi_eta_ordered"])

        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        print(f"  {n_events} events, {keep.sum()} pass m_jj > {args.mjj_cut} GeV "
              f"({100 * keep.mean():.1f}%)")
        cut_dphi.append(obs["dphi"][keep])
        cut_dphi_eta.append(obs["dphi_eta_ordered"][keep])

    new_dphi = np.concatenate(new_dphi)
    new_dphi_eta = np.concatenate(new_dphi_eta)
    cut_dphi = np.concatenate(cut_dphi)
    cut_dphi_eta = np.concatenate(cut_dphi_eta)
    print(f"[new, pooled] {len(new_dphi)} events uncut, {len(cut_dphi)} events "
          f"after m_jj > {args.mjj_cut} GeV")

    for ax, new_arr, cut_arr, old_arr, title in [
        (axes[0], new_dphi, cut_dphi, old_dphi, r"pT-ordered (training convention)"),
        (axes[1], new_dphi_eta, cut_dphi_eta, old_dphi_eta, r"$\eta$-ordered (CP convention)"),
    ]:
        ax.hist(old_arr, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=OLD_COLOR, label="old (Delphes_Data)")
        ax.hist(new_arr, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=NEW_COLOR, label="new, uncut")
        ax.hist(cut_arr, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=CUT_COLOR, label=f"new, hard-scatter $m_{{jj}}$ > {args.mjj_cut:.0f} GeV")
        ax.set_title(rf"Truth $\Delta\phi_{{jj}}$ -- {title}")
        ax.set_xlabel(r"$\Delta\phi_{jj}$")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_xlim(-np.pi, np.pi)

    fig.suptitle("Effect of reinstating the hard-scatter $m_{jj}$ cut on the new-data $\\Delta\\phi_{jj}$ shape "
                 "-- pooled, all 3 scenarios")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nwrote {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True, help="fromPratik_sim_delphes/ (new data)")
    p.add_argument("--old-data-dir", default=None,
                    help="Delphes_Data/ (old data, for reference shape). "
                         "Defaults to the config's data.input_dir.")
    p.add_argument("--mjj-cut", type=float, default=100.0,
                    help="Hard-scatter (truth parton) m_jj cut in GeV to reinstate")
    p.add_argument("--output", required=True)
    p.add_argument("--n-bins", type=int, default=200)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
