from __future__ import annotations
import numpy as np

# ── Angle helpers ──
def delta_phi(phi1, phi2):
    """Wrapped phi difference, result in (-pi, pi]."""
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi

def recover_phi_j2(phi_j1, dphi_jj):
    """phi_j2 = delta_phi(phi_j1, dphi_jj) -- no negative sign, despite appearances."""
    return delta_phi(phi_j1, dphi_jj)

# ── Transform registry: variable-agnostic. Any catalog variable can use any of these,
# without kinematics.py needing to know the variable's name in advance. ──
TRANSFORMS = {
    "identity": (1, lambda v: np.asarray(v)[..., None],
                    lambda a: a[..., 0]),
    "log1p":    (1, lambda v: np.log1p(v)[..., None],
                    lambda a: np.expm1(a[..., 0])),
    "sin_cos":  (2, lambda v: np.stack([np.sin(v), np.cos(v)], axis=-1),
                    lambda a: np.arctan2(a[..., 0], a[..., 1])),
}

def encode_object(values: dict, variable_transforms: list) -> np.ndarray:
    """
    values             : {"pt": ..., "eta": ..., ...} physical quantities
    variable_transforms: [("pt","log1p"), ("eta","identity"), ...] -- from catalog.resolve_object
    returns             : (..., total_width) encoded float32 array, concatenated in list order
    """
    parts = [TRANSFORMS[t][1](values[var]) for var, t in variable_transforms]
    return np.concatenate(parts, axis=-1).astype(np.float32)

def decode_object(feat: np.ndarray, variable_transforms: list) -> dict:
    """Inverse of encode_object -- returns {variable_name: physical_value}."""
    out, offset = {}, 0
    for var, t in variable_transforms:
        width, _, decode_fn = TRANSFORMS[t]
        out[var] = decode_fn(feat[..., offset: offset + width])
        offset += width
    return out

# ── Four-vector construction ──
def four_vector(pt, eta, phi, E=None, mass=None, fixed_mass=0.0):
    """
    Priority: explicit E wins if given. Otherwise derive E from whichever
    mass is available -- a learned per-event mass, or a fixed constant.
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

# kinematics.py — add anywhere in the file, e.g. right after photons_to_higgs

def compute_swap_mask(p_pt: np.ndarray, p_eta: np.ndarray, ordering: str) -> np.ndarray:
    """
    True where slot 0 and slot 1 need swapping so that j1 ends up first,
    per the chosen convention.

    "pt"  -> j1 = higher-pT parton (training default, matches original notebook)
    "eta" -> j1 = the FORWARD parton, i.e. larger *signed* eta -- not |eta|.
             This is the literature convention for a CP-sensitive Delta phi_jj
             (Hankele/Zeppenfeld-style "azimuthal angle correlation" papers,
             ATLAS VBF-Higgs-CP papers): ordering by |eta| or by pT instead
             averages the +Delta phi_jj / -Delta phi_jj contributions
             together and destroys the CP-odd asymmetry.
    """
    if ordering == "pt":
        return p_pt[:, 0] < p_pt[:, 1]
    elif ordering == "eta":
        return p_eta[:, 0] < p_eta[:, 1]
    else:
        raise ValueError(f"Unknown parton_ordering '{ordering}' -- expected 'pt' or 'eta'")

# ── Full-event reconstruction ──
def reconstruct_event(samples: np.ndarray, truth_config: dict) -> dict:
    """
    samples      : (N, truth_dim) unscaled truth array (model output already
                   inverse-transformed out of normalized space)
    truth_config : {
        "objects": {name: {"variable_transforms": [...], "value_type": "energy"|"mass"|"fixed",
                            "fixed_mass": ...}},
        "event":   {"variable_transforms": [...]}   # optional, e.g. dphi_jj
    }
    Returns {"H": (N,4), "j1": (N,4), "j2": (N,4)} of (E, px, py, pz).
    """
    objects = truth_config["objects"]
    event_cfg = truth_config.get("event")

    offset = 0
    decoded = {}
    for name, obj_cfg in objects.items():
        vt = obj_cfg["variable_transforms"]
        width = sum(TRANSFORMS[t][0] for _, t in vt)
        feat = samples[..., offset: offset + width]
        decoded[name] = decode_object(feat, vt)
        offset += width

    decoded_event = {}
    if event_cfg is not None:
        vt = event_cfg["variable_transforms"]
        width = sum(TRANSFORMS[t][0] for _, t in vt)
        feat = samples[..., offset: offset + width]
        decoded_event = decode_object(feat, vt)
        offset += width

    if offset != samples.shape[-1]:
        raise ValueError(f"truth_config sums to {offset} features but samples has {samples.shape[-1]}")

    four_vectors = {}
    phi_j1 = decoded.get("j1", {}).get("phi")

    for name, k in decoded.items():
        obj_cfg = objects[name]
        value_type = obj_cfg.get("value_type", "fixed")
        fixed_mass = obj_cfg.get("fixed_mass", 0.0)

        phi = k.get("phi")
        if phi is None and "dphi_jj" in decoded_event:
            if phi_j1 is None:
                raise ValueError(f"'{name}' needs phi_j1 to recover phi from dphi_jj, "
                                  f"but j1 has no decoded phi.")
            phi = recover_phi_j2(phi_j1, decoded_event["dphi_jj"])

        four_vectors[name] = four_vector(
            k["pt"], k["eta"], phi,
            E=k.get("E"), mass=k.get("mass"), fixed_mass=fixed_mass,
        )

    return four_vectors