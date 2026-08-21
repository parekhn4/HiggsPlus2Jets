"""
Usage
    python scripts/scenario_deviation_from_pooled.py --config configs/no_energy.yaml \
        --data-dir fromPratik_sim_delphes/ --mjj-cut 100.0 --n-bins 200

Checks whether at_1_bt_0's much worse unfolded-vs-truth SSE (see
scripts/compare_unfolded_old_vs_new_truth.py) is explained by its truth-level dphi_jj
shape simply being the most different from the POOLED shape -- since the checkpoint was
trained on all 3 EFT scenarios pooled with no CP label, it can only learn one blended
posterior, so a scenario whose true shape sits furthest from that blend is expected to
fit worst, independent of any per-event modeling limitation. Computes each scenario's
SSE against the pooled (all-3-scenario) truth dphi_jj density, both dphi conventions,
on both the OLD and NEW (m_jj-cut) data for comparison.
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


def report(label: str, per_scenario: dict, bins: np.ndarray) -> None:
    pooled = {
        k: np.concatenate([per_scenario[s][k] for s in SCENARIOS])
        for k in ["dphi", "dphi_eta"]
    }
    pooled_d = dens(pooled["dphi"], bins)
    pooled_de = dens(pooled["dphi_eta"], bins)
    print(f"\n[{label}] SSE(scenario vs pooled-{label}):")
    for s in SCENARIOS:
        d = dens(per_scenario[s]["dphi"], bins)
        de = dens(per_scenario[s]["dphi_eta"], bins)
        print(f"  {s}: pT-ordered={sse(d, pooled_d):.5f}  eta-ordered={sse(de, pooled_de):.5f}  "
              f"n={len(per_scenario[s]['dphi'])}")


def run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    old_data_dir = args.old_data_dir or config["data"]["input_dir"]
    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)

    old_per_scenario = {}
    for scenario in SCENARIOS:
        path = f"{old_data_dir}/{scenario}/tag_1_delphes_events.root"
        native = training.read_native_arrays(path, config)
        extracted = training.select_and_extract(native, config)
        truth_fv = truth_four_vectors_from_extracted(extracted)
        obs = kinematics.build_observables(truth_fv)
        old_per_scenario[scenario] = {"dphi": obs["dphi"], "dphi_eta": obs["dphi_eta_ordered"]}
        print(f"[old, {scenario}] {len(obs['dphi'])} events")

    new_cut_per_scenario = {}
    for scenario in SCENARIOS:
        path = f"{args.data_dir}/{scenario}/merged.root"
        print(f"[{scenario}] reading {path}")
        _, truth_fv = read_and_select(path, config)
        obs = kinematics.build_observables(truth_fv)
        mjj = invariant_mass(truth_fv["j1"], truth_fv["j2"])
        keep = mjj > args.mjj_cut
        new_cut_per_scenario[scenario] = {"dphi": obs["dphi"][keep], "dphi_eta": obs["dphi_eta_ordered"][keep]}
        print(f"  {len(obs['dphi'])} events, {keep.sum()} after m_jj > {args.mjj_cut} GeV")

    report("old", old_per_scenario, bins)
    report(f"new_mjj>{args.mjj_cut:.0f}", new_cut_per_scenario, bins)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--old-data-dir", default=None)
    p.add_argument("--mjj-cut", type=float, default=100.0)
    p.add_argument("--n-bins", type=int, default=200)
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
