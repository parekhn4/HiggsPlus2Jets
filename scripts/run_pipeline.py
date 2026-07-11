"""
Usage
    python scripts/run_pipeline.py --config configs/no_energy.yaml

Runs the full model-development pipeline end to end: preprocess -> train ->
evaluate -> validate_unfolding, writing everything into one auto-named
runs/<date>_<config-name>_<n_blocks>b/ folder (see CLAUDE.md's "Where things
live"). Calls each stage's existing run function directly -- equivalent to
running the commands in README.md's "How to run" by hand, one after another.

Does NOT include inference.py/reduce_posterior.py (unfolding new analysis
data with the resulting checkpoint) -- that's a separate "use an
already-trained model" step, run manually once you're happy with training.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

import training.preprocessing_training as preprocessing_training
import training.train as train_module
import evaluate.evaluate as evaluate_module
import validate_unfolding


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def derive_run_name(config: dict) -> str:
    date = datetime.date.today().isoformat()
    name = config.get("name", "run")
    n_blocks = config.get("model", {}).get("n_blocks", "?")
    return f"{date}_{name}_{n_blocks}b"


def run_pipeline(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or derive_run_name(config)
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    preprocessed_path = run_dir / "preprocessed.h5"
    checkpoint_path = run_dir / "model.pt"

    section("1. Preprocess")
    preprocessing_training.build_and_save(config, str(preprocessed_path))

    section("2. Train")
    train_module.train(argparse.Namespace(
        config=args.config, preprocessed=str(preprocessed_path),
        output=str(checkpoint_path), val_fold=args.val_fold,
    ))

    section("3. Evaluate (closure plots on held-out fold)")
    evaluate_module.run_evaluate(argparse.Namespace(
        checkpoint=str(checkpoint_path), config=args.config,
        preprocessed=str(preprocessed_path), output_dir=str(run_dir / "eval_plots"),
        n_samples=args.n_samples, batch_size=args.batch_size, seed=args.seed,
    ))

    section("4. Validate unfolding (mean vs. single-draw vs. full posterior)")
    validate_unfolding.run_validate(argparse.Namespace(
        checkpoint=str(checkpoint_path), config=args.config,
        preprocessed=str(preprocessed_path), output_dir=str(run_dir / "validation_plots"),
        n_samples=args.n_samples, batch_size=args.batch_size, seed=args.seed,
    ))

    section("DONE")
    print(f"Everything written under {run_dir}/")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run preprocess -> train -> evaluate -> validate_unfolding end to end."
    )
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--val-fold", type=int, default=4, help="Which fold to hold out for validation")
    p.add_argument("--n-samples", type=int, default=200,
                    help="Posterior samples per event for evaluate/validate_unfolding (default: 200)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", help="Override the auto-derived runs/<name>/ folder name")
    p.add_argument("--runs-dir", default="runs", help="Parent directory for run folders (default: runs/)")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
