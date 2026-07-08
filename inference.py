"""
Usage
    python inference.py \\
        --checkpoint best_model.pt \\
        --config configs/no_energy.yaml \\
        --data-dir Delphes_Data/ \\
        --output four_vectors.h5 \\
        --n-samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import kinematics
import preprocessing_inference as inference_prep
from model import build_model_from_config


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────

def load_checkpoint_bundle(checkpoint_path: str, config: dict, device: str) -> dict:
    """
    Load everything needed to run inference from a self-contained
    checkpoint, cross-checking against the runtime config where they
    legitimately overlap (max_jets) rather than trusting either blindly.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_max_jets = checkpoint["max_jets"]
    cfg_max_jets = config["data"]["max_jets"]
    if ckpt_max_jets != cfg_max_jets:
        raise ValueError(
            f"--config's max_jets ({cfg_max_jets}) does not match the checkpoint's "
            f"({ckpt_max_jets}) -- the config was likely edited after training. "
            f"Fix --config to match the checkpoint, don't guess which is right."
        )

    resolved = checkpoint["resolved_config"]
    max_jets = ckpt_max_jets
    reco_dim = kinematics.total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)

    model = build_model_from_config(
        {"model": checkpoint["model_config"]},
        target_dim=truth_dim, context_dim=reco_dim, device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = {
        "x_mean": checkpoint["x_mean"], "x_scale": checkpoint["x_scale"],
        "y_mean": checkpoint["y_mean"], "y_scale": checkpoint["y_scale"],
    }

    return {
        "model": model, "resolved": resolved, "max_jets": max_jets,
        "scaler": scaler, "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
    }


# ──────────────────────────────────────────────────────────────────────────
# Posterior sampling
# ──────────────────────────────────────────────────────────────────────────

def sample_posterior_batch(model, X_reco_scaled: np.ndarray, truth_dim: int,
                            n_samples_per_event: int, device: str,
                            batch_size: int = 512, progress_every: int = 5000) -> np.ndarray:
    """
    Batched posterior sampling. Samples are stored sequentially per
    event: event 0's n_samples_per_event draws first, then event 1's,
    etc. Returns SCALED samples -- caller inverts scaling separately.
    """
    model.eval()
    n_events = len(X_reco_scaled)
    all_samples = []

    with torch.no_grad():
        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            batch_n = end - start

            if start % progress_every == 0:
                print(f"  sampling {start}/{n_events}")

            xb = torch.tensor(X_reco_scaled[start:end], dtype=torch.float32, device=device)
            xb_rep = xb.repeat_interleave(n_samples_per_event, dim=0)

            z = torch.randn(batch_n * n_samples_per_event, truth_dim, device=device)
            samples_scaled = model.inverse(z, xb_rep).cpu().numpy()
            all_samples.append(samples_scaled)

    return np.concatenate(all_samples, axis=0)


# ──────────────────────────────────────────────────────────────────────────
# Output assembly
# ──────────────────────────────────────────────────────────────────────────

def four_vectors_to_dataframe(four_vectors: dict, event_idx: np.ndarray,
                               sample_idx: np.ndarray) -> pd.DataFrame:
    cols = {"event_idx": event_idx, "sample_idx": sample_idx}
    for obj_name, arr in four_vectors.items():
        for i, comp in enumerate(["E", "px", "py", "pz"]):
            cols[f"{obj_name}_{comp}"] = arr[:, i]
    return pd.DataFrame(cols)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def run_inference(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    print(f"loading checkpoint {args.checkpoint}")
    bundle = load_checkpoint_bundle(args.checkpoint, config, device)
    model, resolved, max_jets, scaler = (
        bundle["model"], bundle["resolved"], bundle["max_jets"], bundle["scaler"]
    )
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)
    print(f"  epoch {bundle['epoch']}, val_loss {bundle['val_loss']:.4f}, "
          f"parton_ordering: {resolved['parton_ordering']}")

    scenario_files = inference_prep.discover_scenario_files(args.data_dir, config)
    if not scenario_files:
        print(f"No scenario ROOT files found under {args.data_dir}.", file=sys.stderr)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with pd.HDFStore(args.output, mode="w") as store:
        for scenario_name, path in scenario_files.items():
            print(f"\n[{scenario_name}] reading {path}")
            X_reco, meta = inference_prep.build_reco_features(
                str(path), scenario_name, config, resolved["reco"]
            )
            print(f"  {len(X_reco)} events pass selection")

            X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)

            print(f"  sampling {args.n_samples} posterior draws/event...")
            samples_scaled = sample_posterior_batch(
                model, X_reco_scaled, truth_dim,
                n_samples_per_event=args.n_samples,
                device=device, batch_size=args.batch_size,
            )
            samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)

            four_vectors = kinematics.reconstruct_event(samples, resolved["truth"])

            event_idx = np.repeat(np.arange(len(X_reco)), args.n_samples)
            sample_idx = np.tile(np.arange(args.n_samples), len(X_reco))
            df = four_vectors_to_dataframe(four_vectors, event_idx, sample_idx)

            # merge in the AUX_ metadata (already AUX_-prefixed by
            # build_reco_features), repeated once per posterior sample to
            # match df's event-major row order -- without this, four_vectors.h5
            # has no way to trace an unfolded row back to its source event
            meta_repeated = meta.iloc[event_idx].reset_index(drop=True)
            df = pd.concat([meta_repeated, df], axis=1)

            store.put(scenario_name, df, format="fixed")
            print(f"  wrote {len(df)} rows to {args.output}[{scenario_name}]")

    print(f"\nDone. Output: {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unfold Delphes-simulated Hjj reco events to parton-level four-vectors with a cINN.",
    )
    p.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--config", required=True,
                    help="Config YAML for data-source specifics (scenario paths, selection, tree_name)")
    p.add_argument("--data-dir", required=True, help="Directory containing per-scenario Delphes ROOT files")
    p.add_argument("--output", required=True, help="Output HDF5 path for unfolded four-vectors")
    p.add_argument("--n-samples", type=int, default=500,
                    help="Posterior samples drawn per event (default: 500)")
    p.add_argument("--batch-size", type=int, default=512,
                    help="Reco events per inference batch, before repeat_interleave (default: 512)")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()