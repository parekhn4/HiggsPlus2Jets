"""
Pipeline per scenario file:
  1. read native-named branches (via catalog.resolve_branch_names)
  2. apply selection cuts, pT-order partons
  3. compute physical quantities (photons_to_higgs, dphi_jj, per-jet padding)
  4. encode each object/event block via catalog.resolve_object + kinematics.encode_object
  5. concatenate into X_reco / y_truth, in config's object order
  6. assign a persisted fold column (seeded once, never recomputed)
  7. write one HDF5 key per scenario
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import awkward as ak

import catalog
import kinematics


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


def total_dim(resolved_domain: dict, max_jets: int | None = None) -> int:
    """
    Sum of every object's + the event block's dim, in config order.
    "jet" is a repeated object (up to max_jets copies in the actual
    array), so its contribution is multiplied accordingly -- everything
    else appears exactly once.
    """
    dim = 0
    for name, obj in resolved_domain["objects"].items():
        multiplier = max_jets if (name == "jet" and max_jets is not None) else 1
        dim += obj["dim"] * multiplier
    if "event" in resolved_domain:
        dim += resolved_domain["event"]["dim"]
    return dim


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

    with uproot.open(path) as f:
        arr = f[tree_name].arrays(branch_names, library="ak")

    native_to_branch = dict(zip(NATIVE_BRANCHES, branch_names))
    return {native: arr[branch] for native, branch in native_to_branch.items()}


def select_and_extract(native: dict, config: dict) -> dict:
    """
    Apply selection cuts (>=2 photons, >=2 jets, exactly 1 truth Higgs +
    2 hard partons, Higgs mass window), pT-order the partons, and return
    plain numpy arrays (post-selection) for everything downstream needs.
    """
    sel = config["selection"]

    pid = native["particle_pid"]
    status = native["particle_status"]
    is_higgs = (pid == 25) & (status == 22)
    is_parton = (status == 23) & ((pid == 21) | (abs(pid) <= 5))

    n_higgs = ak.num(native["particle_pt"][is_higgs])
    n_partons = ak.num(native["particle_pt"][is_parton])
    n_photons = ak.num(native["photon_pt"])
    n_jets = ak.num(native["jet_pt"])

    mask = (n_higgs == 1) & (n_partons == 2)
    if sel["require_2_photons"]:
        mask = mask & (n_photons >= 2)
    if sel["require_2_reco_jets"]:
        mask = mask & (n_jets >= 2)

    # ── reco Higgs from diphoton system ──
    pho_pt, pho_eta = native["photon_pt"][mask], native["photon_eta"][mask]
    pho_phi, pho_e = native["photon_phi"][mask], native["photon_E"][mask]

    h_reco_pt, h_reco_eta, h_reco_phi, h_reco_mass, h_reco_E = kinematics.photons_to_higgs(
        ak.to_numpy(pho_pt[:, 0]), ak.to_numpy(pho_eta[:, 0]),
        ak.to_numpy(pho_phi[:, 0]), ak.to_numpy(pho_e[:, 0]),
        ak.to_numpy(pho_pt[:, 1]), ak.to_numpy(pho_eta[:, 1]),
        ak.to_numpy(pho_phi[:, 1]), ak.to_numpy(pho_e[:, 1]),
    )

    mass_ok = np.ones(len(h_reco_mass), dtype=bool)
    if sel["use_mass_window"]:
        center, half_width = sel["mass_window_center"], sel["mass_window_half_width"]
        mass_ok = np.abs(h_reco_mass - center) < half_width

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

    # ── truth partons, pT-ordered (j1 = higher pT) ──
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

    # ── reco jets, padded to max_jets ──
    max_jets = config["data"]["max_jets"]
    jet_pt_ak = native["jet_pt"][mask][mass_ok]
    jet_eta_ak = native["jet_eta"][mask][mass_ok]
    jet_phi_ak = native["jet_phi"][mask][mass_ok]
    jet_mass_ak = native["jet_mass"][mask][mass_ok]

    n_events = len(h_true_pt)

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
        "H_reco": {"pt": h_reco_pt[mass_ok], "eta": h_reco_eta[mass_ok],
                    "phi": h_reco_phi[mass_ok], "mass": h_reco_mass[mass_ok],
                    "E": h_reco_E[mass_ok]},
        "H_truth": {"pt": h_true_pt, "eta": h_true_eta, "phi": h_true_phi,
                     "mass": h_true_mass, "E": h_true_E},
        "j1_truth": {"pt": p_pt[:, 0], "eta": p_eta[:, 0], "phi": p_phi[:, 0],
                      "mass": p_mass[:, 0], "E": p_E[:, 0]},
        "j2_truth": {"pt": p_pt[:, 1], "eta": p_eta[:, 1],
                      "mass": p_mass[:, 1], "E": p_E[:, 1]},  # no standalone phi -- dphi_jj is event-level
        "event_truth": {"dphi_jj": dphi_jj},
        "jet_reco": {"pt": jet_pt, "eta": jet_eta, "phi": jet_phi,
                      "mass": jet_mass, "E": jet_E},
        "event_reco": {"njet": njet},
    }


# ──────────────────────────────────────────────────────────────────────────
# Encoding: physical quantities -> X_reco / y_truth arrays
# ──────────────────────────────────────────────────────────────────────────

def encode_domain(extracted: dict, resolved_domain: dict, domain: str, max_jets: int = None) -> np.ndarray:
    """
    domain: "truth" or "reco". Builds the full feature array by
    concatenating each configured object's encoding, then the event
    block, in the order they appear in resolved_domain["objects"].

    "jet" is handled specially: it's a repeated object (up to max_jets),
    so it gets encoded once per slot and concatenated max_jets times,
    matching the existing reco layout (Higgs, njet, jet_1, ..., jet_N).
    """
    parts = []
    for obj_name, obj_cfg in resolved_domain["objects"].items():
        vt = obj_cfg["variable_transforms"]
        if obj_name == "jet":
            jet_vals = extracted[f"jet_{domain}"]
            for j in range(max_jets):
                slot_vals = {k: v[:, j] for k, v in jet_vals.items()}
                parts.append(kinematics.encode_object(slot_vals, vt))
        else:
            vals = extracted[f"{obj_name}_{domain}"]
            parts.append(kinematics.encode_object(vals, vt))

    if "event" in resolved_domain:
        vt = resolved_domain["event"]["variable_transforms"]
        vals = extracted[f"event_{domain}"]
        parts.append(kinematics.encode_object(vals, vt))

    return np.concatenate(parts, axis=-1)


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

    X_reco = encode_domain(extracted, resolved["reco"], "reco", max_jets=max_jets)
    y_truth = encode_domain(extracted, resolved["truth"], "truth", max_jets=max_jets)

    n_events = extracted["n_events"]
    fold = assign_folds(n_events, config["data"].get("n_folds", 5), config["data"].get("seed", 42))

    df = pd.DataFrame({
        "sample": sample_name,
        "fold": fold,
        "n_reco_jets": extracted["event_reco"]["njet"],
        "reco_higgs_mass": extracted["H_reco"]["mass"],
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