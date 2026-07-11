"""
Usage
    python scripts/unfold_and_average.py \\
        --checkpoint best_model.pt \\
        --config configs/no_energy.yaml \\
        --data-dir Delphes_Data/ \\
        --output-mean four_vectors_mean.h5 \\
        --output-draw four_vectors_draw.h5 \\
        --n-samples 500

One-command path from a checkpoint to a per-event four-vector: runs
inference.py's full posterior sampling, then reduce_posterior.py's
reduction, back to back. Give one or both of --output-mean/--output-draw.
Keeps the intermediate full-samples file by default (at 500 samples/event
it's ~24KB/event -- a few GB for a typical 100k-event scenario, not worth
throwing away) so the expensive sampling step never has to be re-run just
to get at the full posterior later; pass --discard-samples if you really
don't want it.

If you ONLY want --output-draw and never need the mean or the full
posterior, sampling 500/event just to keep 1 is wasteful -- since draws
are i.i.d. (no reweighting needed, see kinematics.select_posterior_draw),
`--n-samples 1` gives the exact same distribution at a fraction of the
compute/storage. The bigger default only pays for itself when you're also
deriving the mean or keeping the full posterior from the same run.

Use inference.py + reduce_posterior.py directly instead of this wrapper if
you need the full per-sample posterior itself (e.g. an optimal observable
computed per-sample rather than from a per-event reduction).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference.inference as inference
import reduce_posterior


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference.py then reduce_posterior.py in one step."
    )
    p.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--config", required=True,
                    help="Config YAML for data-source specifics (scenario paths, selection, tree_name)")
    p.add_argument("--data-dir", required=True, help="Directory containing per-scenario Delphes ROOT files")
    p.add_argument("--output-mean", help="Output HDF5 path for the on-shell per-event mean")
    p.add_argument("--output-draw", help="Output HDF5 path for a single random posterior draw per event")
    p.add_argument("--draw-index", type=int, default=0,
                    help="Which posterior sample index to use for --output-draw (default: 0)")
    p.add_argument("--n-samples", type=int, default=500,
                    help="Posterior samples drawn per event before reduction (default: 500)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-output", metavar="PATH",
                    help="Where to save the full per-sample posterior (inference.py's raw output). "
                         "Defaults to <first output>_samples.h5 next to whichever --output-* is given.")
    p.add_argument("--discard-samples", action="store_true",
                    help="Delete the full per-sample posterior after reduction instead of keeping it "
                         "(default: keep it)")
    return p


def main():
    args = build_arg_parser().parse_args()
    if not args.output_mean and not args.output_draw:
        raise SystemExit("Specify at least one of --output-mean / --output-draw")

    reference_output = args.output_mean or args.output_draw
    samples_path = (
        Path(args.samples_output) if args.samples_output
        else Path(reference_output).with_name(Path(reference_output).stem + "_samples.h5")
    )
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    inference_args = argparse.Namespace(
        checkpoint=args.checkpoint, config=args.config, data_dir=args.data_dir,
        output=str(samples_path), n_samples=args.n_samples,
        batch_size=args.batch_size, seed=args.seed,
    )
    inference.run_inference(inference_args)

    if args.output_mean:
        Path(args.output_mean).parent.mkdir(parents=True, exist_ok=True)
        reduce_posterior.reduce_file(str(samples_path), args.output_mean, method="mean")
        print(f"wrote mean output -> {args.output_mean}")

    if args.output_draw:
        Path(args.output_draw).parent.mkdir(parents=True, exist_ok=True)
        reduce_posterior.reduce_file(str(samples_path), args.output_draw, method="draw", draw_index=args.draw_index)
        print(f"wrote single-draw output -> {args.output_draw}")

    if args.discard_samples:
        samples_path.unlink()
        print(f"discarded full-sample posterior at {samples_path}")
    else:
        print(f"kept full-sample posterior at {samples_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
