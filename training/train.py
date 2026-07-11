from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml

import plotting.plotting as plotting
import training.preprocessing_training as training
from sklearn.preprocessing import StandardScaler
from core.model import build_model_from_config


# ──────────────────────────────────────────────────────────────────────────
# Loss — plain Gaussian NLL for a normalizing flow, confirmed against the
# source notebook: L = 0.5 * ||z||^2 - log|det J|, no physics penalty term.
# ──────────────────────────────────────────────────────────────────────────

def cinn_nll(model, x_truth, x_reco):
    z, log_det = model(x_truth, x_reco)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=1) - log_det)


def compute_eval_nll(model, x_truth: torch.Tensor, x_reco: torch.Tensor, batch_size: int) -> float:
    """
    Full-dataset NLL in eval mode (dropout off), batched to bound memory use.
    Used to get a training-set loss comparable to val_loss (also eval-mode) --
    the raw per-batch training loss during the optimization step is measured
    with dropout ON and is not directly comparable to val_loss on its own.
    """
    model.eval()
    total_loss = 0.0
    n = len(x_truth)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_loss = cinn_nll(model, x_truth[start:end], x_reco[start:end])
            total_loss += batch_loss.item() * (end - start)
    return total_loss / n


# ──────────────────────────────────────────────────────────────────────────
# Scaler fitting — generalized reco/truth scaler fitting.
#
# The selection requires >=2 reco jets, so jet slots 1 and 2 are always real
# data (never zero-padded) -- each gets its own independent fit rather than
# borrowing another slot's statistics. Jet slots 3+ are mostly zero-padding
# (only populated for higher jet-multiplicity events), so fitting them
# independently would blow up their variance; instead they get jet1+jet2
# pooled together (stacked as more samples of the same "generic jet"
# distribution) broadcast onto them -- the best available proxy given
# there's no independently-fittable signal there. Truth scaling has no such
# special case -- plain StandardScaler.
# ──────────────────────────────────────────────────────────────────────────

def resolve_reco_layout(resolved_reco: dict, max_jets: int):
    """
    Column slices for each block in the concatenated X_reco array,
    matching the order kinematics.encode_domain builds it in.
    """
    offset = 0
    non_jet_slices = []
    jet_slot_slices = []
    jet_width = None
    for name, obj in resolved_reco["objects"].items():
        if name == "jet":
            jet_width = obj["dim"]
            for _ in range(max_jets):
                jet_slot_slices.append(slice(offset, offset + jet_width))
                offset += jet_width
        else:
            non_jet_slices.append(slice(offset, offset + obj["dim"]))
            offset += obj["dim"]
    if "event" in resolved_reco:
        w = resolved_reco["event"]["dim"]
        non_jet_slices.append(slice(offset, offset + w))
        offset += w
    return non_jet_slices, jet_width, jet_slot_slices


def fit_reco_scaler(X_train: np.ndarray, resolved_reco: dict, max_jets: int) -> dict:
    """
    Fit on non-jet columns (Higgs, event, ...) plus jet slots 1 and 2
    independently (each real, always-present data); jet slots 3+ get
    jet1+jet2 pooled together broadcast onto them. Returns
    {"mean": (reco_dim,), "scale": (reco_dim,)}.
    """
    non_jet_slices, jet_width, jet_slot_slices = resolve_reco_layout(resolved_reco, max_jets)

    x_mean = np.zeros(X_train.shape[1], dtype=np.float32)
    x_scale = np.ones(X_train.shape[1], dtype=np.float32)

    # non-jet columns + jet slots 1 and 2, each fit on its own real data --
    # combining them in one StandardScaler.fit call is just a code
    # convenience, since StandardScaler computes each column's mean/std
    # independently regardless of what else is in the same call.
    own_fit_slots = jet_slot_slices[:2]
    fit_cols = []
    for s in non_jet_slices:
        fit_cols.extend(range(s.start, s.stop))
    for s in own_fit_slots:
        fit_cols.extend(range(s.start, s.stop))

    scaler = StandardScaler()
    scaler.fit(X_train[:, fit_cols])
    fitted_mean, fitted_scale = scaler.mean_, scaler.scale_

    idx = 0
    for s in non_jet_slices:
        w = s.stop - s.start
        x_mean[s] = fitted_mean[idx: idx + w]
        x_scale[s] = fitted_scale[idx: idx + w]
        idx += w
    for s in own_fit_slots:
        w = s.stop - s.start
        x_mean[s] = fitted_mean[idx: idx + w]
        x_scale[s] = fitted_scale[idx: idx + w]
        idx += w

    # jet slots 3+ get jet1+jet2 pooled (stacked as rows -> more samples of
    # the same generic-jet distribution), not each slot's own fit
    remaining_slots = jet_slot_slices[2:]
    if remaining_slots and own_fit_slots:
        pooled_rows = np.concatenate([X_train[:, s] for s in own_fit_slots], axis=0)
        pooled_scaler = StandardScaler()
        pooled_scaler.fit(pooled_rows)
        for s in remaining_slots:
            x_mean[s] = pooled_scaler.mean_
            x_scale[s] = pooled_scaler.scale_

    return {"mean": x_mean, "scale": x_scale}


def fit_truth_scaler(y_train: np.ndarray) -> dict:
    scaler = StandardScaler()
    scaler.fit(y_train)
    return {"mean": scaler.mean_.astype(np.float32), "scale": scaler.scale_.astype(np.float32)}


# ──────────────────────────────────────────────────────────────────────────
# Data loading — pool all scenarios (model is scenario-agnostic by
# design), split by the persisted fold column, never re-shuffled here.
# ──────────────────────────────────────────────────────────────────────────

def load_pooled_dataset(preprocessed_path: str, reco_dim: int, truth_dim: int) -> pd.DataFrame:
    frames = []
    with pd.HDFStore(preprocessed_path, mode="r") as store:
        for key in store.keys():
            frames.append(store[key])
    df = pd.concat(frames, axis=0, ignore_index=True)

    x_cols = [f"x_{i}" for i in range(reco_dim)]
    y_cols = [f"y_{i}" for i in range(truth_dim)]
    missing = [c for c in x_cols + y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"preprocessed file is missing expected columns: {missing}")

    return df


def split_by_fold(df: pd.DataFrame, val_fold: int, reco_dim: int, truth_dim: int):
    x_cols = [f"x_{i}" for i in range(reco_dim)]
    y_cols = [f"y_{i}" for i in range(truth_dim)]

    train_df = df[df["AUX_fold"] != val_fold]
    val_df = df[df["AUX_fold"] == val_fold]

    X_train = train_df[x_cols].to_numpy(dtype=np.float32)
    y_train = train_df[y_cols].to_numpy(dtype=np.float32)
    X_val = val_df[x_cols].to_numpy(dtype=np.float32)
    y_val = val_df[y_cols].to_numpy(dtype=np.float32)

    return X_train, y_train, X_val, y_val


# ──────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)
        torch.manual_seed(config["data"].get("seed", 42))

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    resolved = training.resolve_config(config)
    max_jets = config["data"]["max_jets"]
    from core.kinematics import total_dim
    reco_dim = total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = total_dim(resolved["truth"], max_jets=max_jets)
    print(f"resolved dims -- reco: {reco_dim}, truth: {truth_dim}, "
          f"parton_ordering: {resolved['parton_ordering']}")

    print(f"loading preprocessed data from {args.preprocessed}")
    df = load_pooled_dataset(args.preprocessed, reco_dim, truth_dim)
    print(f"  pooled dataset: {len(df)} events across {df['AUX_sample'].nunique()} scenarios")

    X_train, y_train, X_val, y_val = split_by_fold(df, args.val_fold, reco_dim, truth_dim)
    print(f"  train: {len(X_train)}  val (fold {args.val_fold}): {len(X_val)}")

    print("fitting scalers on training fold only...")
    x_scaler = fit_reco_scaler(X_train, resolved["reco"], max_jets)
    y_scaler = fit_truth_scaler(y_train)

    def scale_x(X):
        return (X - x_scaler["mean"]) / x_scaler["scale"]

    def scale_y(y):
        return (y - y_scaler["mean"]) / y_scaler["scale"]

    X_train_s = scale_x(X_train)
    y_train_s = scale_y(y_train)
    X_val_s = scale_x(X_val)
    y_val_s = scale_y(y_val)

    model = build_model_from_config(config, target_dim=truth_dim, context_dim=reco_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"].get("lr", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config["training"].get("scheduler_patience", 20),
        factor=0.5,
    )

    X_train_t = torch.tensor(X_train_s, device=device)
    y_train_t = torch.tensor(y_train_s, device=device)
    X_val_t = torch.tensor(X_val_s, device=device)
    y_val_t = torch.tensor(y_val_s, device=device)

    batch_size = config["training"].get("batch_size", 1024)
    max_epochs = config["training"].get("max_epochs", 500)
    early_stop_patience = config["training"].get("early_stop_patience", 40)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history = []
    train_eval_loss_history = []
    val_loss_history = []

    n_train = len(X_train_t)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_losses = []

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = y_train_t[idx], X_train_t[idx]  # (truth, reco) -- model(x_truth, x_reco)

            optimizer.zero_grad()
            loss = cinn_nll(model, xb, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = cinn_nll(model, y_val_t, X_val_t).item()
        train_eval_loss = compute_eval_nll(model, y_train_t, X_train_t, batch_size)

        scheduler.step(val_loss)
        train_loss = float(np.mean(train_losses))
        train_loss_history.append(train_loss)
        train_eval_loss_history.append(train_eval_loss)
        val_loss_history.append(val_loss)
        print(f"epoch {epoch:4d}  train_nll {train_loss:.4f}  "
              f"train_nll(eval) {train_eval_loss:.4f}  val_nll {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(args.output, model, optimizer, scheduler, epoch, val_loss,
                             config, resolved, x_scaler, y_scaler, args.val_fold)
            print(f"  -> saved checkpoint (val_nll {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"early stopping at epoch {epoch} "
                      f"({early_stop_patience} epochs without improvement)")
                break

    loss_plot_path = Path(args.output).with_name(Path(args.output).stem + "_loss_curve.png")
    fig = plotting.plot_loss_curve(train_loss_history, val_loss_history, train_eval_loss_history,
                                    title=f"best val_nll {best_val_loss:.4f}")
    fig.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote loss curve -> {loss_plot_path}")

    print(f"\nDone. Best val_nll: {best_val_loss:.4f}. Checkpoint: {args.output}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss,
                     config, resolved, x_scaler, y_scaler, val_fold):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "model_config": config["model"],
        "resolved_config": resolved,          # truth/reco variable_transforms, value_type,
                                                 # fixed_mass, parton_ordering -- everything
                                                 # inference.py needs to rebuild this exact
                                                 # input/output contract
        "max_jets": config["data"]["max_jets"],  # needed alongside resolved_config to compute dims
        "x_mean": x_scaler["mean"], "x_scale": x_scaler["scale"],
        "y_mean": y_scaler["mean"], "y_scale": y_scaler["scale"],
        "val_fold": val_fold,
        "n_folds": config["data"].get("n_folds", 5),
        "seed": config["data"].get("seed", 42),
    }, path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Hjj unfolding cINN.")
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--preprocessed", required=True, help="Preprocessed HDF5 from preprocessing_training.py")
    p.add_argument("--output", required=True, help="Checkpoint output path (.pt)")
    p.add_argument("--val-fold", type=int, default=4, help="Which fold to hold out for validation")
    return p


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()