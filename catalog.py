# The job of this module is to turn the user's variable selection into a concrete model config

from __future__ import annotations

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

# Translate catalog-native names to a dataset's actual branch names

def resolve_branch_names(native_names: list[str], dataset: str = "delphes") -> list[str]:
    mapping = SCHEMA_MAP[dataset]
    return [mapping[n] for n in native_names]


# Turn (variable, transform) pair into encoded field names that would be understood by decode_kinematics in kinematics.py

def encoded_field_names(variable: str, transform: str) -> list[str]:
    if transform == "identity":
        return [variable]  # like eta
    if transform == "log1p":
        return [f"log_{variable}"]  # these: log_pt, log_mass, log_E, log_njet
    if transform == "sin_cos":
        base = "dphi" if variable == "dphi_jj" else variable
        return [f"{base}_sin", f"{base}_cos"]
    raise ValueError(f"Unknown transform '{transform}' for variable '{variable}'")


# The catalog

CATALOG = {
    "pt":      {"level": "object", "applies_to": ["H", "j1", "j2", "jet"],
                "domains": ["reco", "truth"], "trainable": True, "default_transform": "log1p"},
    "eta":     {"level": "object", "applies_to": ["H", "j1", "j2", "jet"],
                "domains": ["reco", "truth"], "trainable": True, "default_transform": "identity"},
    "phi":     {"level": "object", "applies_to": ["H", "j1", "jet"],
                "domains": ["reco", "truth"], "trainable": True, "default_transform": "sin_cos"},
    "dphi_jj": {"level": "object", "applies_to": ["j2"],
                "domains": ["truth"], "trainable": True, "default_transform": "sin_cos"},
    "mass":    {"level": "object", "applies_to": ["H", "j1", "j2", "jet"],
                "domains": ["reco", "truth"], "trainable": True, "default_transform": "log1p"},
    "E":       {"level": "object", "applies_to": ["H", "jet"],
                "domains": ["reco", "truth"], "trainable": True, "default_transform": "log1p"},
    "njet":    {"level": "event", "applies_to": ["event"],
                "domains": ["reco"], "trainable": True, "default_transform": "log1p"},
}


def validate_selection(variables: list[str], object_name: str, domain: str) -> None:
    for var in variables:
        entry = CATALOG.get(var)
        if entry is None:
            raise ValueError(f"'{var}' is not in the catalog")
        if entry["level"] != "event" and object_name not in entry["applies_to"]:
            raise ValueError(f"'{var}' does not apply to object '{object_name}'")
        if domain not in entry["domains"]:
            raise ValueError(f"'{var}' is not available in domain '{domain}'")


# Resolver

def resolve_object(variables: list[str], object_name: str, domain: str,
                    transforms: dict | None = None) -> dict:
    """
    variables  : e.g. ["pt", "eta", "phi", "mass"] — physical names, config-selected
    transforms : optional {variable: transform_name} override; falls back to each
                 catalog entry's default_transform if not given
    returns    : {"fields": [...], "value_type": "energy"|"mass"|"fixed", "dim": int}
                 — exactly what kinematics.py's make_layout/four_vector expect
    """
    validate_selection(variables, object_name, domain)
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