"""
Usage
    python scripts/reduce_posterior.py --input samples.h5 --output-mean means.h5
    python scripts/reduce_posterior.py --input samples.h5 --output-draw draws.h5
    python scripts/reduce_posterior.py --input samples.h5 --output-mean means.h5 --output-draw draws.h5

Collapses inference.py's full per-event posterior samples into one
four-vector per event, by whichever of the two methods you ask for (give
one or both --output-* paths). Self-contained -- reads only the input file
(value_type/fixed_mass were stored there as attrs by inference.py), no
checkpoint or config needed.

--output-mean : averages pt/eta linearly and phi circularly (angles wrap
    at +-pi, a plain mean is wrong there -- see kinematics.circular_mean),
    then for fixed-mass objects recomputes E from the averaged momenta and
    the fixed mass, so the result is exactly on-shell (mean(E_i) !=
    E(mean(pt), mean(eta)) since E is nonlinear in pt/eta -- averaging
    four-vectors directly would NOT be on-shell even though every
    individual sample is). Cheapest option, but can distort a genuinely
    multimodal observable (e.g. dphi_jj when there's a jet-labeling
    ambiguity) -- see validate_unfolding.py's mean-vs-samples comparison.
--output-draw : keeps a single posterior draw per event (--draw-index,
    default 0) as-is. No re-derivation needed -- every stored sample is
    already an unweighted, self-consistent, on-shell four-vector exactly
    as the model produced it. Unlike the mean, provably preserves the
    correct population-level shape when many events' single draws are
    pooled together (see validate_unfolding.py's closure comparison).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

import core.kinematics as kinematics


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


def single_draw_four_vector(fv: np.ndarray, draw_index: int = 0) -> np.ndarray:
    """fv: (n_events, n_samples, 4) -> (n_events, 4), one posterior draw per event."""
    return fv[:, draw_index, :]


def reduce_file(input_path: str, output_path: str, method: str, draw_index: int = 0) -> None:
    """method: "mean" or "draw"."""
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
                if method == "mean":
                    value_type = str(ds.attrs["value_type"])
                    fixed_mass = float(ds.attrs["fixed_mass"])
                    reduced = average_four_vector_samples(fv, value_type, fixed_mass)
                elif method == "draw":
                    reduced = single_draw_four_vector(fv, draw_index)
                else:
                    raise ValueError(f"Unknown method '{method}' -- expected 'mean' or 'draw'")
                unfolded_out.create_dataset(name, data=reduced)

            meta_out = out_grp.create_group("meta")
            for key, ds in in_grp["meta"].items():
                meta_out.create_dataset(key, data=ds[...])

            print(f"[{scenario}] reduced ({method}) {n_samples} samples/event -> {n_events} events")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collapse inference.py's full posterior samples into one four-vector per event."
    )
    p.add_argument("--input", required=True, help="Nested HDF5 written by inference.py")
    p.add_argument("--output-mean", help="Output HDF5 path for the on-shell per-event mean")
    p.add_argument("--output-draw", help="Output HDF5 path for a single random posterior draw per event")
    p.add_argument("--draw-index", type=int, default=0,
                    help="Which posterior sample index to use for --output-draw (default: 0)")
    return p


def main():
    args = build_arg_parser().parse_args()
    if not args.output_mean and not args.output_draw:
        raise SystemExit("Specify at least one of --output-mean / --output-draw")

    if args.output_mean:
        Path(args.output_mean).parent.mkdir(parents=True, exist_ok=True)
        reduce_file(args.input, args.output_mean, method="mean")
        print(f"Done. Mean output: {args.output_mean}")

    if args.output_draw:
        Path(args.output_draw).parent.mkdir(parents=True, exist_ok=True)
        reduce_file(args.input, args.output_draw, method="draw", draw_index=args.draw_index)
        print(f"Done. Single-draw output: {args.output_draw}")


if __name__ == "__main__":
    main()
