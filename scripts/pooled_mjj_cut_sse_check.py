"""
Usage
    python scripts/pooled_mjj_cut_sse_check.py --config configs/no_energy.yaml \
        --data-dir fromPratik_sim_delphes/ --mjj-cut 100.0 --n-bins 40

Pooled (all 3 scenarios) sum-of-squared-density-difference check: old truth dphi_jj vs.
new truth dphi_jj (uncut) vs. new truth dphi_jj (m_jj > --mjj-cut cut reinstated), both
pT-ordered and eta-ordered conventions. This is the FIRST version of this check, originally
run as an inline `python -c` one-liner (never saved to a file) right after establishing the
m_jj-cut hypothesis -- saved here as a real, rerunnable script per the 2026-08-20 review
that flagged unsaved inline checks as a trust/reproducibility gap (see CLAUDE.md/session
history). scripts/compare_old_new_truth_dphi.py is the properly-saved, per-scenario+pooled
successor to this same check; this script exists specifically to reproduce the original
pooled-only numbers exactly for cross-verification, not as a new analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

import core.kinematics as kinematics
import training.preprocessing_training as training
from scripts.build_forMarcel_handoff import truth_four_vectors_from_extracted
from scripts.evaluate_pratik_data import read_and_select, SCENARIOS
from scripts.check_mjj_cut_effect import invariant_mass


def dens(a: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(a, bins=bins, density=True)
    return counts


def sse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    old_data_dir = args.old_data_dir or config["data"]["input_dir"]

    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)

    old_dphi, old_dphi_eta = [], []
    for scenario in SCENARIOS:
        path = f"{old_data_dir}/{scenario}/tag_1_delphes_events.root"
        native = training.read_native_arrays(path, config)
        extracted = training.select_and_extract(native, config)
        truth_fv = truth_four_vectors_from_extracted(extracted)
        obs = kinematics.build_observables(truth_fv)
        old_dphi.append(obs["dphi"]); old_dphi_eta.append(obs["dphi_eta_ordered"])
    old_dphi = np.concatenate(old_dphi); old_dphi_eta = np.concatenate(old_dphi_eta)

    new_dphi, new_dphi_eta, cut_dphi, cut_dphi_eta = [], [], [], []
    for scenario in SCENARIOS:
        path = f"{args.data_dir}/{scenario}/merged.root"
        extracted, truth_fv = read_and_select(path, config)
        obs = kinematics.build_observables(truth_fv)
        new_dphi.append(obs["dphi"]); new_dphi_eta.append(obs["dphi_eta_ordered"])
        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        cut_dphi.append(obs["dphi"][keep]); cut_dphi_eta.append(obs["dphi_eta_ordered"][keep])
    new_dphi = np.concatenate(new_dphi); new_dphi_eta = np.concatenate(new_dphi_eta)
    cut_dphi = np.concatenate(cut_dphi); cut_dphi_eta = np.concatenate(cut_dphi_eta)

    old_d, old_de = dens(old_dphi, bins), dens(old_dphi_eta, bins)
    new_d, new_de = dens(new_dphi, bins), dens(new_dphi_eta, bins)
    cut_d, cut_de = dens(cut_dphi, bins), dens(cut_dphi_eta, bins)

    print(f"n_bins={args.n_bins}  mjj_cut={args.mjj_cut}")
    print(f"old: {len(old_dphi)}  new uncut: {len(new_dphi)}  new cut: {len(cut_dphi)}")
    print("pT-ordered  SSE(new_uncut vs old) =", sse(new_d, old_d))
    print("pT-ordered  SSE(new_cut   vs old) =", sse(cut_d, old_d))
    print("eta-ordered SSE(new_uncut vs old) =", sse(new_de, old_de))
    print("eta-ordered SSE(new_cut   vs old) =", sse(cut_de, old_de))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--old-data-dir", default=None)
    p.add_argument("--mjj-cut", type=float, default=100.0)
    p.add_argument("--n-bins", type=int, default=40)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
