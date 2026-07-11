"""
Usage
    python inference/inference.py \\
        --checkpoint best_model.pt \\
        --config configs/no_energy.yaml \\
        --data-dir Delphes_Data/ \\
        --output four_vectors.h5 \\
        --n-samples 500

Writes ONE self-contained HDF5 file with the full posterior: every sampled
four-vector for every event, nothing averaged or reduced. Per scenario:

    /{scenario}/four_vectors/reco/{H,j1,j2}      (n_events, 4) of (E,px,py,pz)
    /{scenario}/four_vectors/unfolded/{H,j1,j2}  (n_events, n_samples, 4)
    /{scenario}/meta/{event_id,...}               (n_events,) -- one row per event

reco and unfolded are both plain four-vectors (config-agnostic -- any
pt/eta/phi/dphi_jj/etc. derived quantity can be computed from either one the
same way, see kinematics.build_observables). Each unfolded dataset carries
"value_type" and "fixed_mass" attrs, so downstream reduction (see
reduce_posterior.py) doesn't need the checkpoint or config to know which
objects are on a fixed mass shell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import yaml

import core.kinematics as kinematics
import inference.preprocessing_inference as inference_prep
from core.model import build_model_from_config


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

def write_scenario_group(h5file: h5py.File, scenario_name: str, reco_fv: dict, unfolded_fv: dict,
                          meta, resolved_truth: dict, n_events: int, n_samples: int) -> None:
    grp = h5file.create_group(scenario_name)

    fv_grp = grp.create_group("four_vectors")
    fv_grp.attrs["components"] = "E, px, py, pz"

    reco_grp = fv_grp.create_group("reco")
    for name, fv in reco_fv.items():
        reco_grp.create_dataset(name, data=fv.astype(np.float32))

    unfolded_grp = fv_grp.create_group("unfolded")
    for name, fv in unfolded_fv.items():
        ds = unfolded_grp.create_dataset(name, data=fv.reshape(n_events, n_samples, 4).astype(np.float32))
        obj_cfg = resolved_truth["objects"][name]
        ds.attrs["value_type"] = obj_cfg.get("value_type", "fixed")
        ds.attrs["fixed_mass"] = obj_cfg.get("fixed_mass", 0.0)

    meta_grp = grp.create_group("meta")
    for col in meta.columns:
        meta_grp.create_dataset(col, data=meta[col].to_numpy())


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

    with h5py.File(args.output, "w") as h5file:
        for scenario_name, path in scenario_files.items():
            print(f"\n[{scenario_name}] reading {path}")
            X_reco, meta = inference_prep.build_reco_features(
                str(path), scenario_name, config, resolved["reco"]
            )
            n_events = len(X_reco)
            print(f"  {n_events} events pass selection")

            X_reco_scaled = inference_prep.apply_reco_scaling(X_reco, scaler)

            print(f"  sampling {args.n_samples} posterior draws/event...")
            torch.manual_seed(args.seed)
            samples_scaled = sample_posterior_batch(
                model, X_reco_scaled, truth_dim,
                n_samples_per_event=args.n_samples,
                device=device, batch_size=args.batch_size,
            )
            samples = inference_prep.invert_truth_scaling(samples_scaled, scaler)

            unfolded_fv = kinematics.reconstruct_event(samples, resolved["truth"])
            reco_fv = kinematics.reco_four_vectors(X_reco, resolved["reco"], max_jets)

            write_scenario_group(h5file, scenario_name, reco_fv, unfolded_fv, meta.reset_index(drop=True),
                                  resolved["truth"], n_events=n_events, n_samples=args.n_samples)
            print(f"  wrote {n_events} events x {args.n_samples} samples -> {args.output}[{scenario_name}]")

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
    p.add_argument("--seed", type=int, default=42, help="Seed for posterior sampling (default: 42)")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()