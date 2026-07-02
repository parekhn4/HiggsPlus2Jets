from __future__ import annotations
from kinematics import TRANSFORMS

# The job of this module is to turn the user's variable selection into a concrete model config. It maps a dataset's actual branch names onto the catalog-native names every compute/resolve function in this file works with. The schema map below can be updated for data or any other MC if need be 

SCHEMA_MAP = {
    "delphes": {
        "jet_pt": "Jet.PT", "jet_eta": "Jet.Eta", "jet_phi": "Jet.Phi", "jet_mass": "Jet.Mass",
        "photon_pt": "Photon.PT", "photon_eta": "Photon.Eta",
        "photon_phi": "Photon.Phi", "photon_E": "Photon.E",
        "particle_pid": "Particle.PID", "particle_status": "Particle.Status",
        "particle_pt": "Particle.PT", "particle_eta": "Particle.Eta",
        "particle_phi": "Particle.Phi", "particle_mass": "Particle.Mass",
    },
}

def resolve_branch_names(native_names: list[str], dataset: str = "delphes") -> list[str]:
    # Translate catalog-native names to a dataset's actual ROOT branch names.
    mapping = SCHEMA_MAP[dataset]
    return [mapping[n] for n in native_names]


# The catalog: physical quantities only. Encoding (log1p, sin_cos, ...) is transform metadata, kept separate from the variable's identity, so the same physical quantity can be encoded differently by different configs.

# "level": "object" quantities belong to a specific H/j1/j2/jet and get sliced out of that object's own block.
#  "level": "event" quantities are not owned by any single object (njet, dphi_jj, the optimal observable) 

CATALOG = {
    "pt": {
        "level": "object", "applies_to": ["H", "j1", "j2", "jet"],
        "domains": ["reco", "truth"], "default_transform": "log1p",
    },
    "eta": {
        "level": "object", "applies_to": ["H", "j1", "j2", "jet"],
        "domains": ["reco", "truth"], "default_transform": "identity",
    },
    "phi": {
        "level": "object", "applies_to": ["H", "j1", "jet"],  
        "domains": ["reco", "truth"], "default_transform": "sin_cos",
    },
    "mass": {
        "level": "object", "applies_to": ["H", "j1", "j2", "jet"],
        "domains": ["reco", "truth"], "default_transform": "log1p",
    },
    "E": {
        "level": "object", "applies_to": ["H", "j1", "j2", "jet"],
        "domains": ["reco", "truth"], "default_transform": "log1p",
    },
    "dphi_jj": {
        "level": "event", "applies_to": ["event"],
        "domains": ["truth"], "default_transform": "sin_cos",
    },
    "njet": {
        "level": "event", "applies_to": ["event"],
        "domains": ["reco"], "default_transform": "log1p",
    },
    "optimal_observable": {
        "level": "event", "applies_to": ["event"],
        "domains": ["reco", "truth"], "default_transform": "identity",
    },
}

def validate_selection(variables: list[str], target: str, domain: str) -> None:
    for var in variables:
        entry = CATALOG.get(var)
        if entry is None:
            raise ValueError(f"'{var}' is not in the catalog")
        if target not in entry["applies_to"]:
            raise ValueError(f"'{var}' does not apply to '{target}'")
        if domain not in entry["domains"]:
            raise ValueError(f"'{var}' is not available in domain '{domain}'")
        if not entry["trainable"]:
            raise ValueError(
                f"'{var}' is not trainable (plotting/analysis-only) and cannot be "
                f"selected in a truth/reco config block."
            )

# The one resolver: catalog selection -> the field/value_type shape kinematics.py's make_layout/decode_kinematics/four_vector already expect.
# Works for both physics objects (H, j1, j2, jet) and "event" — event-level selections just never end up with an E/mass, so value_type stays "fixed" and is simply unused for those.

def resolve_object(variables: list[str], target: str, domain: str,
                    transforms: dict | None = None) -> dict:
    validate_selection(variables, target, domain)
    transforms = transforms or {}

    variable_transforms = [
        (var, transforms.get(var, CATALOG[var]["default_transform"]))
        for var in variables
    ]
    dim = sum(_TRANSFORM_WIDTHS[t] for _, t in variable_transforms)

    value_type = "energy" if "E" in variables else "mass" if "mass" in variables else "fixed"
    return {"variable_transforms": variable_transforms, "value_type": value_type, "dim": dim}