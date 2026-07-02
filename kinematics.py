"""
Pure kinematics math shared by both model variants (no_energy, with_energy).

No I/O, no config-file reading, no ROOT/torch imports — just numpy. Everything
here operates on plain arrays and small dicts, so it's identical code whether
called from data.py (training), preprocessing.py (inference reco-building), or
inference.py (posterior sample -> four-vector).

Design: field layout is name-keyed, not positional. A model variant's object
just lists which named fields it has (e.g. ["log_pt", "eta", "phi_sin",
"phi_cos", "log_E"]); make_layout() turns that into a {name: index} dict, and
decode_kinematics() reads by name. Nothing here ever assumes "feature 0 is
always pt" — that assumption only ever existed implicitly before, and broke
exactly when with_energy added a 5th slot with a different meaning per object.

The one thing that's still genuinely positional: j2's phi is never stored
directly, in either model. Its sin/cos slot stores delta_phi_jj =
delta_phi(phi_j1, phi_j2), so recovering true phi_j2 requires phi_j1 as an
external input. That coupling is real physics, not an artifact of a bad
layout, so it's handled explicitly in reconstruct_event() rather than papered
over.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Angle helpers
# ──────────────────────────────────────────────────────────────────────────

def delta_phi(phi1, phi2):
    """Wrapped phi difference, result in (-pi, pi]."""
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def recover_phi_j2(phi_j1, dphi_jj):
    """
    j2's phi slot stores delta_phi_jj = delta_phi(phi_j1, phi_j2), so the
    actual phi_j2 is recovered as delta_phi(phi_j1, delta_phi_jj) —
    note: no negative sign, despite delta_phi being antisymmetric-looking.
    """
    return delta_phi(phi_j1, dphi_jj)


# ──────────────────────────────────────────────────────────────────────────
# Named field layout — order-independent by construction
# ──────────────────────────────────────────────────────────────────────────

def make_layout(fields: list[str]) -> dict[str, int]:
    """
    ["log_pt", "eta", "phi_sin", "phi_cos", "log_E"]
      -> {"log_pt": 0, "eta": 1, "phi_sin": 2, "phi_cos": 3, "log_E": 4}

    The config's field list decides the actual array order (it has to —
    the array is a real tensor); this function is what makes that the
    *only* place order is decided. Everything downstream reads by name
    through the returned dict instead of hardcoding an index.
    """
    return {name: i for i, name in enumerate(fields)}


def decode_kinematics(feat: np.ndarray, layout: dict[str, int]) -> dict:
    """
    feat   : (..., len(layout)) array
    layout : from make_layout()
    returns: dict with whichever of {pt, eta, phi, dphi, E, mass} are
             present in this layout — decode is driven entirely by which
             named fields exist, nothing assumed.
    """
    out = {}
    if "log_pt" in layout:
        out["pt"] = np.expm1(feat[..., layout["log_pt"]])
    if "eta" in layout:
        out["eta"] = feat[..., layout["eta"]]
    if "phi_sin" in layout and "phi_cos" in layout:
        out["phi"] = np.arctan2(feat[..., layout["phi_sin"]], feat[..., layout["phi_cos"]])
    if "dphi_sin" in layout and "dphi_cos" in layout:
        out["dphi"] = np.arctan2(feat[..., layout["dphi_sin"]], feat[..., layout["dphi_cos"]])
    if "log_E" in layout:
        out["E"] = np.expm1(feat[..., layout["log_E"]])
    if "log_mass" in layout:
        out["mass"] = np.expm1(feat[..., layout["log_mass"]])
    return out


# ──────────────────────────────────────────────────────────────────────────
# Four-vector construction
# ──────────────────────────────────────────────────────────────────────────

def four_vector(pt, eta, phi, E=None, mass=None, fixed_mass=0.0):
    """
    (pt, eta, phi, [E or mass or fixed_mass]) -> (E, px, py, pz).

    Priority: explicit E wins if given (with_energy's Higgs). Otherwise
    derive E from whichever mass is available — a learned per-event mass
    (with_energy's jets) or a fixed constant (no_energy, any object).
    """
    if E is None:
        m = mass if mass is not None else fixed_mass
        E = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack([E, px, py, pz], axis=-1)


def photons_to_higgs(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2):
    """Combine two photon 4-vectors into a reconstructed Higgs (pt, eta, phi, mass, E)."""
    px1 = pt1 * np.cos(phi1); py1 = pt1 * np.sin(phi1); pz1 = pt1 * np.sinh(eta1)
    px2 = pt2 * np.cos(phi2); py2 = pt2 * np.sin(phi2); pz2 = pt2 * np.sinh(eta2)
    e = e1 + e2; px = px1 + px2; py = py1 + py2; pz = pz1 + pz2
    pt = np.sqrt(px ** 2 + py ** 2)
    phi = np.arctan2(py, px)
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    eta = 0.5 * np.log((p + pz + 1e-12) / (p - pz + 1e-12))
    mass = np.sqrt(np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))
    return pt, eta, phi, mass, e


# ──────────────────────────────────────────────────────────────────────────
# Full-event reconstruction — handles the j1 -> j2 phi coupling explicitly
# ──────────────────────────────────────────────────────────────────────────

def reconstruct_event(samples: np.ndarray, truth_config: dict) -> dict:
    """
    samples      : (N, truth_dim) unscaled truth array (model output, already
                   inverse-transformed out of normalized space)
    truth_config : config["truth"] — {"objects": {name: {"fields": [...],
                   "value_type": "energy"|"mass"|"fixed", "fixed_mass": ...}},
                   ...}

    Returns {"H": (N,4), "j1": (N,4), "j2": (N,4)} of (E, px, py, pz).

    Slices are computed from each object's own field list length, in the
    order objects appear in truth_config["objects"] — that ordering has to
    match how the truth vector was actually built (data.py), same as it
    always implicitly has, but the *within-object* layout is name-driven
    regardless.
    """
    objects = truth_config["objects"]

    offset = 0
    decoded = {}
    for name, obj_cfg in objects.items():
        width = len(obj_cfg["fields"])
        layout = make_layout(obj_cfg["fields"])
        feat = samples[..., offset: offset + width]
        decoded[name] = decode_kinematics(feat, layout)
        offset += width

    if offset != samples.shape[-1]:
        raise ValueError(
            f"truth_config objects sum to {offset} features but samples has "
            f"{samples.shape[-1]} — config/model mismatch."
        )

    four_vectors = {}
    phi_j1 = decoded["j1"].get("phi") if "j1" in decoded else None

    for name, k in decoded.items():
        obj_cfg = objects[name]
        value_type = obj_cfg.get("value_type", "fixed")
        fixed_mass = obj_cfg.get("fixed_mass", 0.0)

        phi = k.get("phi")
        if phi is None and "dphi" in k:
            if phi_j1 is None:
                raise ValueError(f"'{name}' needs phi_j1 to recover its phi from dphi, "
                                  f"but no 'j1' object with a decoded phi was found.")
            phi = recover_phi_j2(phi_j1, k["dphi"])

        four_vectors[name] = four_vector(
            k["pt"], k["eta"], phi,
            E=k.get("E"), mass=k.get("mass"), fixed_mass=fixed_mass,
        )

    return four_vectors
