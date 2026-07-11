from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import uproot
import awkward as ak
import yaml
import argparse
import datetime

import core.catalog as catalog
import core.kinematics as kinematics
import inference.preprocessing_inference as preprocessing_inference


# ──────────────────────────────────────────────────────────────────────────
# Config resolution — done once, reused across all scenario files, and
# exactly what gets saved into a self-contained checkpoint later.
# ──────────────────────────────────────────────────────────────────────────

def resolve_config(config: dict) -> dict:
    """
    Turn the config's catalog selections (truth/reco objects + event) into
    resolved {"variable_transforms": [...], "value_type": ..., "dim": ...}
    blocks, via catalog.resolve_object. This resolved structure is what
    kinematics.encode_object/decode_object/reconstruct_event consume, and
    what a checkpoint should persist (closes the "what was this model
    trained on" gap from the scaler problem).
    """
    resolved = {
        "truth": {"objects": {}}, "reco": {"objects": {}},
        "parton_ordering": config["data"].get("parton_ordering", "pt"),
    }

    for domain in ("truth", "reco"):
        domain_cfg = config.get(domain, {})
        for obj_name, obj_cfg in domain_cfg.get("objects", {}).items():
            r = catalog.resolve_object(obj_cfg["variables"], obj_name, domain)
            if "fixed_mass" in obj_cfg:
                r["fixed_mass"] = obj_cfg["fixed_mass"]
            resolved[domain]["objects"][obj_name] = r

        event_cfg = domain_cfg.get("event")
        if event_cfg is not None:
            resolved[domain]["event"] = catalog.resolve_object(
                event_cfg["variables"], "event", domain
            )

    return resolved


# ──────────────────────────────────────────────────────────────────────────
# ROOT reading
# ──────────────────────────────────────────────────────────────────────────

NATIVE_BRANCHES = [
    "jet_pt", "jet_eta", "jet_phi", "jet_mass",
    "photon_pt", "photon_eta", "photon_phi", "photon_E",
    "particle_pid", "particle_status",
    "particle_pt", "particle_eta", "particle_phi", "particle_mass",
]


def read_native_arrays(path: str, config: dict) -> dict:
    """ROOT -> {native_name: awkward array}, via the dataset's schema map."""
    dataset = config.get("dataset", "delphes")
    branch_names = catalog.resolve_branch_names(NATIVE_BRANCHES, dataset=dataset)
    tree_name = config["data"]["tree_name"]
    max_events = config["data"].get("max_events")


    with uproot.open(path) as f:
        arr = f[tree_name].arrays(branch_names, entry_stop=max_events, library="ak")

    native_to_branch = dict(zip(NATIVE_BRANCHES, branch_names))
    return {native: arr[branch] for native, branch in native_to_branch.items()}


def select_and_extract(native: dict, config: dict) -> dict:
    """
    Apply selection cuts (>=2 photons, >=2 jets, exactly 1 truth Higgs +
    2 hard partons, Higgs mass window), pT- or eta-order the partons
    (per config["data"]["parton_ordering"]), and return plain numpy
    arrays (post-selection) for everything downstream needs.

    The reco half (Higgs-from-photons, jet padding, njet) is NOT
    duplicated here -- it's the exact same computation
    preprocessing_inference.py's extract_reco_quantities already does for
    inference, so this just combines the truth-level cuts with
    preprocessing_inference.reco_selection_mask's cuts and delegates.
    """
    sel = config["selection"]

    pid = native["particle_pid"]
    status = native["particle_status"]
    is_higgs = (pid == 25) & (status == 22)
    is_parton = (status == 23) & ((pid == 21) | (abs(pid) <= 5))

    n_higgs = ak.num(native["particle_pt"][is_higgs])
    n_partons = ak.num(native["particle_pt"][is_parton])

    mask = (n_higgs == 1) & (n_partons == 2)
    mask = mask & preprocessing_inference.reco_selection_mask(native, config)

    reco = preprocessing_inference.extract_reco_quantities(native, mask, config)
    mass_ok = reco["mass_ok"]

    # ── truth Higgs ──
    h_pt = native["particle_pt"][is_higgs][mask]
    h_eta = native["particle_eta"][is_higgs][mask]
    h_phi = native["particle_phi"][is_higgs][mask]
    h_mass = native["particle_mass"][is_higgs][mask]
    h_true_pt = ak.to_numpy(h_pt[:, 0])[mass_ok]
    h_true_eta = ak.to_numpy(h_eta[:, 0])[mass_ok]
    h_true_phi = ak.to_numpy(h_phi[:, 0])[mass_ok]
    h_true_mass = ak.to_numpy(h_mass[:, 0])[mass_ok]
    h_true_E = np.sqrt((h_true_pt * np.cosh(h_true_eta)) ** 2 + h_true_mass ** 2)

    # ── truth partons, ordered per config["data"]["parton_ordering"] ──
    p_pt = ak.to_numpy(native["particle_pt"][is_parton][mask])[mass_ok]
    p_eta = ak.to_numpy(native["particle_eta"][is_parton][mask])[mass_ok]
    p_phi = ak.to_numpy(native["particle_phi"][is_parton][mask])[mass_ok]
    p_mass = ak.to_numpy(native["particle_mass"][is_parton][mask])[mass_ok]

    ordering = config["data"].get("parton_ordering", "pt")
    swap = kinematics.compute_swap_mask(p_pt, p_eta, ordering)
    for a in (p_pt, p_eta, p_phi, p_mass):
        tmp = a[swap, 0].copy()
        a[swap, 0] = a[swap, 1]
        a[swap, 1] = tmp

    p_E = np.sqrt((p_pt * np.cosh(p_eta)) ** 2 + p_mass ** 2)
    dphi_jj = kinematics.delta_phi(p_phi[:, 0], p_phi[:, 1])

    return {
        "n_events": reco["n_events"],
        "event_id": reco["event_id"],
        "H_reco": reco["H_reco"],
        "jet_reco": reco["jet_reco"],
        "event_reco": reco["event_reco"],
        "H_truth": {"pt": h_true_pt, "eta": h_true_eta, "phi": h_true_phi,
                     "mass": h_true_mass, "E": h_true_E},
        "j1_truth": {"pt": p_pt[:, 0], "eta": p_eta[:, 0], "phi": p_phi[:, 0],
                      "mass": p_mass[:, 0], "E": p_E[:, 0]},
        "j2_truth": {"pt": p_pt[:, 1], "eta": p_eta[:, 1],
                      "mass": p_mass[:, 1], "E": p_E[:, 1]},  # no standalone phi -- dphi_jj is event-level
        "event_truth": {"dphi_jj": dphi_jj},
    }


# ──────────────────────────────────────────────────────────────────────────
# Folds — assigned once, persisted, never recomputed
# ──────────────────────────────────────────────────────────────────────────

def assign_folds(n_events: int, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.arange(n_events) % n_folds
    rng.shuffle(folds)
    return folds.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────
# Top-level: one scenario file -> one HDF5 key
# ──────────────────────────────────────────────────────────────────────────

def build_scenario(path: str, sample_name: str, config: dict, resolved: dict) -> pd.DataFrame:
    native = read_native_arrays(path, config)
    extracted = select_and_extract(native, config)
    max_jets = config["data"]["max_jets"]

    X_reco = kinematics.encode_domain(extracted, resolved["reco"], "reco", max_jets=max_jets)
    y_truth = kinematics.encode_domain(extracted, resolved["truth"], "truth", max_jets=max_jets)

    n_events = extracted["n_events"]
    fold = assign_folds(n_events, config["data"].get("n_folds", 5), config["data"].get("seed", 42))

    df = pd.DataFrame({
        "AUX_sample": sample_name,
        "AUX_event_id": extracted["event_id"],
        "AUX_fold": fold,
        "AUX_n_reco_jets": extracted["event_reco"]["njet"],
        "AUX_reco_higgs_mass": extracted["H_reco"]["mass"],
    })
    for i in range(X_reco.shape[1]):
        df[f"x_{i}"] = X_reco[:, i]
    for i in range(y_truth.shape[1]):
        df[f"y_{i}"] = y_truth[:, i]

    return df


def build_and_save(config: dict, output_path: str) -> None:
    resolved = resolve_config(config)
    base = Path(config["data"]["input_dir"])

    with pd.HDFStore(output_path, mode="w") as store:
        for name, rel_path in config["data"]["scenarios"].items():
            full_path = base / rel_path
            if not full_path.exists():
                print(f"skipping {name}: {full_path} not found")
                continue
            df = build_scenario(str(full_path), name, config, resolved)
            store.put(name, df, format="fixed")
            print(f"[{name}] wrote {len(df)} events -> {output_path}[{name}]")

    # sidecar snapshot: exact config that produced this h5, plus when --
    # answers "what config built this file" without needing to trust
    # whatever configs/*.yaml happens to say later, since that file can
    # change after the fact
    
    snapshot = dict(config)  # copy, don't mutate the original
    snapshot["_snapshot_created_at"] = datetime.datetime.now().isoformat()
    snapshot_path = f"{output_path}.config.yaml"
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)
    print(f"wrote config snapshot -> {snapshot_path}")

# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the training-time preprocessed HDF5 (reco + truth, per scenario, with fold column)."
    )
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--output", required=True, help="Output HDF5 path")
    p.add_argument("--seed", type=int, default=42, help="Seed for posterior sampling (default: 42)")

    return p


def main():
    args = build_arg_parser().parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    build_and_save(config, args.output)

if __name__ == "__main__":
    main()