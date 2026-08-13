from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import awkward as ak

import core.catalog as catalog
import core.kinematics as kinematics


NATIVE_RECO_BRANCHES = [
    "jet_pt", "jet_eta", "jet_phi", "jet_mass",
    "photon_pt", "photon_eta", "photon_phi", "photon_E",
]


def read_native_reco_arrays(path: str, config: dict) -> dict:
    """ROOT -> {native_name: awkward array}, reco branches only."""
    dataset = config.get("dataset", "delphes")
    branch_names = catalog.resolve_branch_names(NATIVE_RECO_BRANCHES, dataset=dataset)
    tree_name = config["data"]["tree_name"]
    max_events = config["data"].get("max_events")

    with uproot.open(path) as f:
        arr = f[tree_name].arrays(branch_names, entry_stop=max_events, library="ak")

    native_to_branch = dict(zip(NATIVE_RECO_BRANCHES, branch_names))
    return {native: arr[branch] for native, branch in native_to_branch.items()}


def apply_object_quality_cuts(native: dict, config: dict) -> dict:
    """
    Filter jets/photons down to quality-passing objects only (pT and |eta|
    acceptance cuts) *before* anything downstream sees them -- so
    "require >=2 jets"/"require >=2 photons" and the Higgs/jet construction
    all operate on quality objects, not on however many raw reconstructed
    objects Delphes happened to report regardless of how soft or forward
    they are. Returns a new dict (native itself is not mutated); truth-only
    keys (e.g. particle_*, present in the training-side native dict) pass
    through untouched. Config keys default to "no cut" if absent, so a
    config that doesn't set them behaves exactly as before this existed.
    """
    sel = config["selection"]
    jet_pt_min = sel.get("jet_pt_min", 0.0)
    jet_eta_max = sel.get("jet_eta_max", float("inf"))
    photon_pt_min = sel.get("photon_pt_min", 0.0)
    photon_eta_max = sel.get("photon_eta_max", float("inf"))

    out = dict(native)

    jet_ok = (native["jet_pt"] > jet_pt_min) & (np.abs(native["jet_eta"]) < jet_eta_max)
    for key in ("jet_pt", "jet_eta", "jet_phi", "jet_mass"):
        if key in out:
            out[key] = out[key][jet_ok]

    photon_ok = (native["photon_pt"] > photon_pt_min) & (np.abs(native["photon_eta"]) < photon_eta_max)
    for key in ("photon_pt", "photon_eta", "photon_phi", "photon_E"):
        if key in out:
            out[key] = out[key][photon_ok]

    return out


def reco_selection_mask(native: dict, config: dict):
    """
    >=2 photons, >=2 jets -- the reco-only cuts. Returns an awkward boolean
    mask. Assumes native has already been through apply_object_quality_cuts
    if pT/eta acceptance cuts are configured -- this function only counts,
    it doesn't filter objects itself.
    """
    sel = config["selection"]
    n_photons = ak.num(native["photon_pt"])
    n_jets = ak.num(native["jet_pt"])

    mask = n_jets >= 0  # always-true awkward boolean array, same length as events
    if sel["require_2_photons"]:
        mask = mask & (n_photons >= 2)
    if sel["require_2_reco_jets"]:
        mask = mask & (n_jets >= 2)
    return mask


def extract_reco_quantities(native: dict, mask, config: dict) -> dict:
    """
    Given native reco arrays and an event mask (already combined by the
    caller with whatever else it needs -- e.g. preprocessing_training.py
    ANDs in truth cuts), build the reco Higgs (from photons), padded jet
    arrays, and njet. Also applies the Higgs mass-window cut and returns
    the resulting mass_ok boolean array plus n_events, so callers with
    additional (e.g. truth) arrays can align to the same final selection.
    """
    sel = config["selection"]

    pho_pt, pho_eta = native["photon_pt"][mask], native["photon_eta"][mask]
    pho_phi, pho_e = native["photon_phi"][mask], native["photon_E"][mask]

    h_pt, h_eta, h_phi, h_mass, h_E = kinematics.photons_to_higgs(
        ak.to_numpy(pho_pt[:, 0]), ak.to_numpy(pho_eta[:, 0]),
        ak.to_numpy(pho_phi[:, 0]), ak.to_numpy(pho_e[:, 0]),
        ak.to_numpy(pho_pt[:, 1]), ak.to_numpy(pho_eta[:, 1]),
        ak.to_numpy(pho_phi[:, 1]), ak.to_numpy(pho_e[:, 1]),
    )

    mass_ok = np.ones(len(h_mass), dtype=bool)
    if sel["use_mass_window"]:
        center, half_width = sel["mass_window_center"], sel["mass_window_half_width"]
        mass_ok = np.abs(h_mass - center) < half_width

    # original event index in the ROOT file, tracked through both selection
    # stages -- lets a surviving row be traced back to "which event in the
    # file", which otherwise disappears once rows get pooled/reindexed
    mask_np = ak.to_numpy(mask)
    event_id = np.where(mask_np)[0][mass_ok]

    max_jets = config["data"]["max_jets"]
    jet_pt_ak = native["jet_pt"][mask][mass_ok]
    jet_eta_ak = native["jet_eta"][mask][mass_ok]
    jet_phi_ak = native["jet_phi"][mask][mass_ok]
    jet_mass_ak = native["jet_mass"][mask][mass_ok]

    n_events = int(mass_ok.sum())

    jet_pt = np.zeros((n_events, max_jets), dtype=np.float32)
    jet_eta = np.zeros((n_events, max_jets), dtype=np.float32)
    jet_phi = np.zeros((n_events, max_jets), dtype=np.float32)
    jet_mass = np.zeros((n_events, max_jets), dtype=np.float32)
    jet_E = np.zeros((n_events, max_jets), dtype=np.float32)

    for j in range(max_jets):
        has_j = ak.num(jet_pt_ak) > j
        idx = np.where(ak.to_numpy(has_j))[0]
        if len(idx) == 0:
            continue
        pt_j = ak.to_numpy(jet_pt_ak[has_j][:, j])
        eta_j = ak.to_numpy(jet_eta_ak[has_j][:, j])
        phi_j = ak.to_numpy(jet_phi_ak[has_j][:, j])
        mass_j = ak.to_numpy(jet_mass_ak[has_j][:, j])
        jet_pt[idx, j] = pt_j
        jet_eta[idx, j] = eta_j
        jet_phi[idx, j] = phi_j
        jet_mass[idx, j] = mass_j
        jet_E[idx, j] = np.sqrt((pt_j * np.cosh(eta_j)) ** 2 + mass_j ** 2)

    njet = ak.to_numpy(ak.num(jet_pt_ak)).astype(np.float32)

    return {
        "n_events": n_events,
        "mass_ok": mass_ok,
        "event_id": event_id,
        "H_reco": {"pt": h_pt[mass_ok], "eta": h_eta[mass_ok], "phi": h_phi[mass_ok],
                    "mass": h_mass[mass_ok], "E": h_E[mass_ok]},
        "jet_reco": {"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "mass": jet_mass, "E": jet_E},
        "event_reco": {"njet": njet},
    }


def build_reco_features(path: str, sample_name: str, config: dict,
                         resolved_reco: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Full inference-time pipeline: ROOT -> selection -> encoded X_reco.

    resolved_reco must come from preprocessing_training.resolve_config(config)["reco"]
    (or a checkpoint's saved resolved config) -- never refit or reselected
    here, same reasoning as the scaler: reco encoding must match exactly
    what the model was trained on, not whatever this file's config says
    in isolation.
    """
    native = read_native_reco_arrays(path, config)
    native = apply_object_quality_cuts(native, config)
    mask = reco_selection_mask(native, config)
    extracted = extract_reco_quantities(native, mask, config)

    max_jets = config["data"]["max_jets"]
    X_reco = kinematics.encode_domain(extracted, resolved_reco, "reco", max_jets=max_jets)

    meta = pd.DataFrame({
        "AUX_sample": sample_name,
        "AUX_event_id": extracted["event_id"],
        "AUX_n_reco_jets": extracted["event_reco"]["njet"],
        "AUX_reco_higgs_mass": extracted["H_reco"]["mass"],
    })

    return X_reco, meta


def discover_scenario_files(data_dir: str, config: dict) -> dict:
    base = Path(data_dir)
    found = {}
    for name, rel_path in config["data"]["scenarios"].items():
        full_path = base / rel_path
        if full_path.exists():
            found[name] = full_path
    return found


def load_scaler(path: str) -> dict:
    data = np.load(path)
    return {
        "x_mean": data["x_mean"].astype(np.float32),
        "x_scale": data["x_scale"].astype(np.float32),
        "y_mean": data["y_mean"].astype(np.float32),
        "y_scale": data["y_scale"].astype(np.float32),
    }


def apply_reco_scaling(X_reco: np.ndarray, scaler: dict) -> np.ndarray:
    return ((X_reco - scaler["x_mean"]) / scaler["x_scale"]).astype(np.float32)


def invert_truth_scaling(y_scaled: np.ndarray, scaler: dict) -> np.ndarray:
    return (y_scaled * scaler["y_scale"] + scaler["y_mean"]).astype(np.float32)