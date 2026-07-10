"""
Usage
    python reduce_posterior.py --input four_vectors.h5 --output four_vectors_mean.h5

Collapses inference.py's full per-event posterior samples into one
four-vector per event. Self-contained -- reads only the input file
(value_type/fixed_mass were stored there as attrs by inference.py), no
checkpoint or config needed.

Averages pt/eta linearly and phi circularly (angles wrap at +-pi, a plain
mean is wrong there -- see kinematics.circular_mean), then for fixed-mass
objects recomputes E from the averaged momenta and the fixed mass, so the
result is exactly on-shell (mean(E_i) != E(mean(pt), mean(eta)) since E is
nonlinear in pt/eta -- averaging four-vectors directly would NOT be
on-shell even though every individual sample is). Objects with a learned
mass/energy (value_type != "fixed") have no physics constraint to
re-derive E from, so their sampled E is just averaged directly instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import kinematics


def average_four_vector_samples(fv: np.ndarray, value_type: str, fixed_mass: float) -> np.ndarray:
    """fv: (n_events, n_samples, 4) of (E, px, py, pz) -> (n_events, 4)."""
    E, px, py = fv[..., 0], fv[..., 1], fv[..., 2]
    pt = np.sqrt(px ** 2 + py ** 2)
    eta = kinematics.four_vector_eta(fv)
    phi = kinematics.four_vector_phi(fv)

    pt_mean = np.mean(pt, axis=1)
    eta_mean = np.mean(eta, axis=1)
    phi_mean = kinematics.circular_mean(phi, axis=1)

    if value_type == "fixed":
        return kinematics.four_vector(pt_mean, eta_mean, phi_mean, fixed_mass=fixed_mass)
    return kinematics.four_vector(pt_mean, eta_mean, phi_mean, E=np.mean(E, axis=1))


def reduce_file(input_path: str, output_path: str) -> None:
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        for scenario in fin.keys():
            in_grp = fin[scenario]
            out_grp = fout.create_group(scenario)

            fv_out = out_grp.create_group("four_vectors")
            fv_out.attrs["components"] = "E, px, py, pz"

            # reco is already one four-vector/event -- copy through unchanged
            reco_out = fv_out.create_group("reco")
            for name, ds in in_grp["four_vectors"]["reco"].items():
                reco_out.create_dataset(name, data=ds[...])

            unfolded_out = fv_out.create_group("unfolded")
            n_events = n_samples = None
            for name, ds in in_grp["four_vectors"]["unfolded"].items():
                fv = ds[...]
                n_events, n_samples = fv.shape[0], fv.shape[1]
                value_type = str(ds.attrs["value_type"])
                fixed_mass = float(ds.attrs["fixed_mass"])
                unfolded_out.create_dataset(name, data=average_four_vector_samples(fv, value_type, fixed_mass))

            meta_out = out_grp.create_group("meta")
            for key, ds in in_grp["meta"].items():
                meta_out.create_dataset(key, data=ds[...])

            print(f"[{scenario}] reduced {n_samples} samples/event -> {n_events} on-shell mean events")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collapse inference.py's full posterior samples into one four-vector per event."
    )
    p.add_argument("--input", required=True, help="Nested HDF5 written by inference.py")
    p.add_argument("--output", required=True, help="Output HDF5 path for per-event mean four-vectors")
    return p


def main():
    args = build_arg_parser().parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    reduce_file(args.input, args.output)
    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
