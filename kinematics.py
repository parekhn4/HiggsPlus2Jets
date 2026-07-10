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
def decode_truth_samples(samples: np.ndarray, truth_config: dict) -> tuple[dict, dict]:
    """
    samples      : (N, truth_dim) unscaled truth array (model output already
                   inverse-transformed out of normalized space)
    truth_config : {
        "objects": {name: {"variable_transforms": [...], "value_type": "energy"|"mass"|"fixed",
                            "fixed_mass": ...}},
        "event":   {"variable_transforms": [...]}   # optional, e.g. dphi_jj
    }
    Returns (decoded, decoded_event): per-object and event-level physical
    quantities, *before* four-vectors are built -- this is the point at
    which posterior samples should be averaged (see average_posterior_samples),
    since averaging already-built four-vectors is not the same thing.
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

    return decoded, decoded_event


def four_vectors_from_decoded(decoded: dict, decoded_event: dict, objects: dict) -> dict:
    """Build {"H": (N,4), "j1": (N,4), "j2": (N,4)} of (E, px, py, pz) from decode_truth_samples' output."""
    four_vectors = {}
    phi_j1 = decoded.get("j1", {}).get("phi")

    for name, k in decoded.items():
        obj_cfg = objects[name]
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


def reconstruct_event(samples: np.ndarray, truth_config: dict) -> dict:
    """Decode + build four-vectors for every sample, one-to-one (no averaging)."""
    decoded, decoded_event = decode_truth_samples(samples, truth_config)
    return four_vectors_from_decoded(decoded, decoded_event, truth_config["objects"])


def circular_mean(angles: np.ndarray, axis: int) -> np.ndarray:
    """Mean of angles handling the +-pi wraparound, via the mean resultant vector."""
    return np.arctan2(np.mean(np.sin(angles), axis=axis), np.mean(np.cos(angles), axis=axis))


def _average_decoded_block(decoded_vals: dict, variable_transforms: list,
                            n_events: int, n_samples: int) -> dict:
    transform_of = dict(variable_transforms)
    out = {}
    for var, val in decoded_vals.items():
        val = val.reshape(n_events, n_samples)
        out[var] = circular_mean(val, axis=1) if transform_of.get(var) == "sin_cos" else np.mean(val, axis=1)
    return out


def average_posterior_samples(samples: np.ndarray, truth_config: dict,
                               n_events: int, n_samples: int) -> dict:
    """
    Collapse n_samples posterior draws/event into one four-vector per event,
    averaging the *physical quantities* (pt, eta linearly; phi/dphi_jj
    circularly, since they wrap at +-pi) before four-vectors are built --
    not averaging already-built (E, px, py, pz) vectors afterward.

    This matters for any fixed-mass object: E is derived there as
    sqrt(pt^2*cosh(eta)^2 + fixed_mass^2), which is nonlinear in pt/eta, so
    mean(E_i) != E(mean(pt), mean(eta)) -- averaging four-vectors directly
    would leave the mean event off-shell even though every individual
    sample is exactly on-shell. Averaging pt/eta first and building E fresh
    from the fixed mass guarantees the returned event is on-shell for every
    fixed-mass object (confirmed against truth for the no-energy model).
    Objects with a learned "mass" or "energy" value_type aren't
    constrained this way -- their averaged E/mass is whatever the network
    produced, since there's no physics constraint to re-derive it from.
    """
    decoded, decoded_event = decode_truth_samples(samples, truth_config)
    objects = truth_config["objects"]
    event_cfg = truth_config.get("event")

    averaged = {
        name: _average_decoded_block(decoded[name], objects[name]["variable_transforms"], n_events, n_samples)
        for name in decoded
    }
    averaged_event = (
        _average_decoded_block(decoded_event, event_cfg["variable_transforms"], n_events, n_samples)
        if event_cfg is not None else {}
    )
    return four_vectors_from_decoded(averaged, averaged_event, objects)

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
                parts.append(encode_object(slot_vals, vt))
        else:
            vals = extracted[f"{obj_name}_{domain}"]
            parts.append(encode_object(vals, vt))

    if "event" in resolved_domain:
        vt = resolved_domain["event"]["variable_transforms"]
        vals = extracted[f"event_{domain}"]
        parts.append(encode_object(vals, vt))

    return np.concatenate(parts, axis=-1)


def decode_domain(X: np.ndarray, resolved_domain: dict, domain: str, max_jets: int = None) -> dict:
    """
    Inverse of encode_domain: a flat encoded array -> the same
    {"H_<domain>": {...}, "jet_<domain>": {...}, "event_<domain>": {...}}
    shape that data.py/preprocessing build before encoding. Repeated
    "jet" slots are decoded individually then stacked into (N, max_jets)
    arrays per physical quantity, matching how extract_reco_quantities
    already shapes jet arrays.
    """
    offset = 0
    out = {}
    for obj_name, obj_cfg in resolved_domain["objects"].items():
        vt = obj_cfg["variable_transforms"]
        width = sum(TRANSFORMS[t][0] for _, t in vt)
        if obj_name == "jet":
            slot_dicts = []
            for _ in range(max_jets):
                feat = X[..., offset:offset + width]
                slot_dicts.append(decode_object(feat, vt))
                offset += width
            keys = slot_dicts[0].keys()
            out[f"jet_{domain}"] = {k: np.stack([sd[k] for sd in slot_dicts], axis=1) for k in keys}
        else:
            feat = X[..., offset:offset + width]
            out[f"{obj_name}_{domain}"] = decode_object(feat, vt)
            offset += width

    if "event" in resolved_domain:
        vt = resolved_domain["event"]["variable_transforms"]
        width = sum(TRANSFORMS[t][0] for _, t in vt)
        feat = X[..., offset:offset + width]
        out[f"event_{domain}"] = decode_object(feat, vt)
        offset += width

    return out


def four_vector_eta(fv: np.ndarray) -> np.ndarray:
    """eta from a (..., 4) (E, px, py, pz) four-vector."""
    px, py, pz = fv[..., 1], fv[..., 2], fv[..., 3]
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    return 0.5 * np.log((p + pz + 1e-12) / (p - pz + 1e-12))


def four_vector_phi(fv: np.ndarray) -> np.ndarray:
    """phi from a (..., 4) (E, px, py, pz) four-vector."""
    return np.arctan2(fv[..., 2], fv[..., 1])


def eta_ordered_dphi_jj(fv_a: np.ndarray, fv_b: np.ndarray) -> np.ndarray:
    """
    The CP-sensitive signed Delta phi_jj: order the two jets by *signed*
    rapidity (not |eta|, not pT -- see literature note in
    compute_swap_mask), forward jet first, then take
    delta_phi(phi_forward, phi_backward). Computed directly from
    four-vectors so it's correct regardless of which convention
    (parton_ordering: pt or eta) was used to label j1/j2 at training
    time -- this is always re-derived fresh, not read off training labels.
    """
    eta_a, eta_b = four_vector_eta(fv_a), four_vector_eta(fv_b)
    phi_a, phi_b = four_vector_phi(fv_a), four_vector_phi(fv_b)

    a_is_forward = eta_a >= eta_b
    phi_forward = np.where(a_is_forward, phi_a, phi_b)
    phi_backward = np.where(a_is_forward, phi_b, phi_a)
    return delta_phi(phi_forward, phi_backward)