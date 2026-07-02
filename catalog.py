from __future__ import annotations

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

# Turn (variable, transform) pair into encoded field names that would be understood by decode_kinematics in kinematics.py.
def encoded_field_names(variable: str, transform: str) -> list[str]:
    if transform == "identity":
        return [variable]  # e.g. "eta"
    if transform == "log1p":
        return [f"log_{variable}"]  # log_pt, log_mass, log_E, log_njet
    if transform == "sin_cos":
        base = "dphi" if variable == "dphi_jj" else variable
        return [f"{base}_sin", f"{base}_cos"]
    raise ValueError(f"Unknown transform '{transform}' for variable '{variable}'")

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

    fields = []
    for var in variables:
        transform = transforms.get(var, CATALOG[var]["default_transform"])
        fields.extend(encoded_field_names(var, transform))

    if "E" in variables:
        value_type = "energy"
    elif "mass" in variables:
        value_type = "mass"
    else:
        value_type = "fixed"

    return {"fields": fields, "value_type": value_type, "dim": len(fields)}