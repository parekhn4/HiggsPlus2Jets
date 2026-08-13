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
# Energy-conservation penalty (torch-native, differentiable) — for configs
# where an object's truth variables include a free "E" instead of a fixed
# mass (see configs/with_energy.yaml). Nothing constrains a freely-predicted
# E to stay consistent with pt/eta and the object's real physical mass, so
# this softly penalizes g = E^2 - |p|^2 - m^2 (should be ~0 for an on-shell
# particle) on a genuine posterior draw generated during training.
#
# Deliberately NOT built on core/kinematics.py's decode/four_vector (numpy,
# used by every other pipeline stage) -- duplicated here in torch so
# gradients flow back into the model. Automatically a no-op (returns 0) for
# any config where no truth object has a free "E" (no_energy, higgs_only,
# jets_only) -- nothing needs to check for that separately.
# ──────────────────────────────────────────────────────────────────────────

# log1p's decode clamps its input to 15 before expm1 -- expm1(15) ~= 3.3e6
# GeV, already absurd for a physical pt/E (LHC beam energy itself is ~6500
# GeV), and float32 expm1 overflows to inf around ~88. Confirmed root cause
# of a real NaN divergence (2026-08-11, configs/with_energy.yaml, epoch 54):
# an untrained/early-training network can output a wild pre-expm1 value,
# expm1 overflows to inf, inf propagates through energy_conservation_penalty
# (E^2 -> inf -> loss -> inf), and .backward() through inf yields nan
# gradients that optimizer.step() then bakes into the model permanently.
# Only used by decode_truth_torch (the penalty's own torch-native decode,
# not core/kinematics.py's numpy one used everywhere else), so this can't
# affect anything outside the penalty computation.
_TORCH_DECODERS = {
    "identity": lambda a: a[..., 0],
    "log1p": lambda a: torch.expm1(torch.clamp(a[..., 0], max=15.0)),
    "sin_cos": lambda a: torch.atan2(a[..., 0], a[..., 1]),
}
_TORCH_WIDTHS = {"identity": 1, "log1p": 1, "sin_cos": 2}


def decode_truth_torch(x_truth_scaled: torch.Tensor, resolved_truth: dict,
                        y_mean: torch.Tensor, y_scale: torch.Tensor) -> tuple[dict, dict]:
    """
    Differentiable counterpart to kinematics.decode_truth_samples, torch-only
    and only unpacking what the energy-conservation penalty needs (pt, eta,
    phi, E -- not mass/other vars). x_truth_scaled: (batch, truth_dim) in
    the model's internal (standardized) space, e.g. straight out of
    model.inverse(). Mirrors kinematics.decode_domain's exact object/event
    ordering and offset bookkeeping -- keep in sync by hand if that changes.
    """
    x = x_truth_scaled * y_scale + y_mean  # unscale to the physical encoding
    objects = resolved_truth["objects"]

    offset = 0
    decoded = {}
    for name, obj_cfg in objects.items():
        vt = obj_cfg["variable_transforms"]
        width = sum(_TORCH_WIDTHS[t] for _, t in vt)
        feat = x[..., offset:offset + width]
        decoded[name] = {}
        sub_offset = 0
        for var, t in vt:
            w = _TORCH_WIDTHS[t]
            decoded[name][var] = _TORCH_DECODERS[t](feat[..., sub_offset:sub_offset + w])
            sub_offset += w
        offset += width

    decoded_event = {}
    if "event" in resolved_truth:
        vt = resolved_truth["event"]["variable_transforms"]
        width = sum(_TORCH_WIDTHS[t] for _, t in vt)
        feat = x[..., offset:offset + width]
        sub_offset = 0
        for var, t in vt:
            w = _TORCH_WIDTHS[t]
            decoded_event[var] = _TORCH_DECODERS[t](feat[..., sub_offset:sub_offset + w])
            sub_offset += w

    return decoded, decoded_event


def energy_conservation_penalty(model, x_reco_scaled: torch.Tensor, resolved_truth: dict,
                                 y_mean: torch.Tensor, y_scale: torch.Tensor,
                                 target_masses: dict) -> torch.Tensor:
    """
    Mean over the batch of sum_over_objects(g^2), g = E^2 - |p|^2 - m^2, on a
    fresh posterior draw (z ~ N(0,1) through model.inverse -- a genuine,
    differentiable sample, not the ground truth, which is already exactly
    on-shell by construction and so isn't a useful training signal here).
    m is each object's *real physical* mass from target_masses (config's
    physics.m_H/m_j) -- not a fixed_mass on the object, which doesn't exist
    once "E" is a free target. Objects without "E" in their variables are
    skipped (their E is still exactly on-shell by construction regardless,
    see kinematics.four_vector). Caller applies the penalty weight k.
    """
    batch = x_reco_scaled.shape[0]
    y_dim = y_mean.shape[0]
    z_gen = torch.randn(batch, y_dim, device=x_reco_scaled.device)
    x_gen = model.inverse(z_gen, x_reco_scaled)

    decoded, decoded_event = decode_truth_torch(x_gen, resolved_truth, y_mean, y_scale)

    phi_j1 = decoded.get("j1", {}).get("phi")
    total = None
    for name, k in decoded.items():
        if "E" not in k:
            continue
        pt, eta, E = k["pt"], k["eta"], k["E"]
        phi = k.get("phi")
        if phi is None:
            # j2 has no standalone phi -- recovered from j1's phi + dphi_jj,
            # same as kinematics.recover_phi_j2. No modular wrap needed: cos/sin
            # are 2*pi-periodic, so the unwrapped difference gives identical px/py.
            if phi_j1 is None or "dphi_jj" not in decoded_event:
                raise ValueError(f"'{name}' has no phi and none can be recovered from dphi_jj")
            phi = phi_j1 - decoded_event["dphi_jj"]
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        # eta decodes via plain "identity" (no transform), so unlike pt/E it isn't
        # bounded by the log1p clamp above -- an untrained/early-training network can
        # still output an absurd eta (confirmed: -31.8, decode_truth_torch never caught
        # it), and torch.sinh overflows float32 around the same magnitude expm1 does.
        # Clamped here directly, at the one place eta feeds into something
        # overflow-prone, rather than broadening the log1p fix to cover it.
        pz = pt * torch.sinh(torch.clamp(eta, min=-15.0, max=15.0))
        m2 = target_masses.get(name, 0.0) ** 2
        g = E ** 2 - (px ** 2 + py ** 2 + pz ** 2) - m2
        # Backstop, independent of the eta clamp above: bounds g^2 to at most 1e30
        # (g clamped to +-1e15) regardless of which input produced an extreme g --
        # real, healthy g values observed in training are O(1e5-1e8), so this only
        # ever engages on genuine outliers, same margin-of-absurdity reasoning as the
        # other clamps here.
        g = torch.clamp(g, min=-1e15, max=1e15)
        term = g ** 2
        total = term if total is None else total + term

    if total is None:
        return torch.zeros((), device=x_reco_scaled.device)
    return total.mean()


# ──────────────────────────────────────────────────────────────────────────
# Residual/event-by-event agreement penalty (torch-native, differentiable) --
# NLL alone only trains the flow to match the *population-level* posterior
# shape; it has no term that rewards a single posterior draw landing close to
# that event's actual paired truth. This adds one: a genuine posterior draw
# (z ~ N(0,1) through model.inverse, same construction as the energy penalty
# above) is compared directly to the paired ground truth in scaled feature
# space -- no physical decode needed (unlike the energy penalty), so this
# works for any config regardless of which objects/variables it trains on.
#
# Real risk, not hypothetical: this term structurally pulls generated samples
# toward the *mean* of the posterior (least-squares-to-a-fixed-target always
# does), which is the exact mechanism already caught elsewhere in this
# project causing a spurious dphi_jj=0 bump under --output-mean on genuinely
# multimodal events (see CLAUDE.md "Things that will bite you"). Too large a
# weight here risks quietly narrowing posterior width / collapsing multimodal
# events even though NLL itself is otherwise healthy. Unlike
# ENERGY_PENALTY_WEIGHT, this weight is a CLI flag (--residual-penalty-weight,
# default 0.0) rather than a module constant -- it needs to default to "off"
# so every existing config/run is unaffected unless explicitly opted in.
# ──────────────────────────────────────────────────────────────────────────

def residual_agreement_penalty(model, x_reco_scaled: torch.Tensor,
                                x_truth_scaled: torch.Tensor) -> torch.Tensor:
    """
    Mean over the batch of ||x_gen - x_truth||^2 (scaled feature space), where
    x_gen = model.inverse(z, x_reco_scaled) for a fresh z ~ N(0,1) -- a
    genuine, differentiable posterior draw, paired against that same event's
    real truth. Caller applies the penalty weight.
    """
    z_gen = torch.randn_like(x_truth_scaled)
    x_gen = model.inverse(z_gen, x_reco_scaled)
    return torch.mean(torch.sum((x_gen - x_truth_scaled) ** 2, dim=1))


def energy_score_penalty(model, x_reco_scaled: torch.Tensor,
                          x_truth_scaled: torch.Tensor) -> torch.Tensor:
    """
    Energy score: E[||X_gen - X_truth||] - 0.5*E[||X_gen - X_gen'||], two
    independent posterior draws (X_gen, X_gen') per event plus the paired
    truth, all in scaled feature space -- same z~N(0,1)->model.inverse
    construction as residual_agreement_penalty, just two draws instead of
    one. Unlike raw MSE, this is a strictly proper scoring rule: its unique
    minimum (over the space of predictive distributions) is the true
    posterior itself, not a point mass at the posterior mean -- the second
    (spread) term penalizes an overconfident/collapsed posterior directly,
    which is what makes this the documented preferred choice over
    residual_agreement_penalty for dphi_jj-style multimodal targets (see
    CLAUDE.md's "dphi_jj residual/loss discussion"). Costs two inverse
    passes instead of one (three total forward/inverse passes per training
    step alongside cinn_nll's forward pass).
    """
    z1 = torch.randn_like(x_truth_scaled)
    z2 = torch.randn_like(x_truth_scaled)
    x_gen1 = model.inverse(z1, x_reco_scaled)
    x_gen2 = model.inverse(z2, x_reco_scaled)
    accuracy = torch.norm(x_gen1 - x_truth_scaled, dim=1)
    spread = torch.norm(x_gen1 - x_gen2, dim=1)
    return torch.mean(accuracy - 0.5 * spread)


# k is a small fixed constant, not a config hyperparameter, by design --
# tune by hand and watch the loss curve. g is in raw GeV^2 units
# (E^2 - |p|^2 - m^2); measured on configs/with_energy.yaml (untrained network,
# 3-epoch smoke test): raw mean(g^2) ~1.67e18 at epoch 0 (E decodes via expm1,
# so a wildly uninitialized flow output blows up exponentially), dropping
# ~100x to ~1e16 by epoch 2 from the NLL term alone. 1e-8 was tried first and
# was nowhere near small enough -- it gave a weighted contribution ~5.6e9 vs.
# the NLL's own O(10) scale, completely swamping the primary objective. This
# value instead targets the weighted penalty landing near the NLL's own scale
# at the *worst* (epoch-0) observed magnitude -- re-check both numbers in the
# training log (printed every epoch) on a real run and retune if the penalty
# still dominates, or if it shrinks to irrelevance once g^2 itself has dropped
# far below its untrained-epoch-0 scale.
ENERGY_PENALTY_WEIGHT = 1e-17


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

    # for the energy-conservation penalty -- see its docstring. Automatically
    # a no-op (returns 0) for any config where no truth object has a free "E".
    y_mean_t = torch.tensor(y_scaler["mean"], device=device)
    y_scale_t = torch.tensor(y_scaler["scale"], device=device)
    physics_cfg = config.get("physics", {})
    target_masses = {
        "H": physics_cfg.get("m_H", 0.0),
        "j1": physics_cfg.get("m_j", 0.0),
        "j2": physics_cfg.get("m_j", 0.0),
    }

    batch_size = config["training"].get("batch_size", 1024)
    max_epochs = config["training"].get("max_epochs", 500)
    early_stop_patience = config["training"].get("early_stop_patience", 40)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history = []
    train_eval_loss_history = []
    val_loss_history = []

    n_train = len(X_train_t)

    residual_weight = getattr(args, "residual_penalty_weight", 0.0) or 0.0
    energy_score_weight = getattr(args, "energy_score_weight", 0.0) or 0.0
    if residual_weight and energy_score_weight:
        raise ValueError("--residual-penalty-weight and --energy-score-weight are two "
                          "different formulations of the same idea -- use one, not both, "
                          "so results stay attributable to a single change.")

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_losses, train_nlls, train_penalties, train_residuals = [], [], [], []

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = y_train_t[idx], X_train_t[idx]  # (truth, reco) -- model(x_truth, x_reco)

            optimizer.zero_grad()
            nll = cinn_nll(model, xb, yb)
            penalty = energy_conservation_penalty(model, yb, resolved["truth"],
                                                    y_mean_t, y_scale_t, target_masses)
            loss = nll + ENERGY_PENALTY_WEIGHT * penalty
            if residual_weight:
                residual = residual_agreement_penalty(model, yb, xb)
                loss = loss + residual_weight * residual
                train_residuals.append(residual.item())
            elif energy_score_weight:
                residual = energy_score_penalty(model, yb, xb)
                loss = loss + energy_score_weight * residual
                train_residuals.append(residual.item())
            loss.backward()
            # defense-in-depth alongside the log1p clamp above -- caps any
            # remaining large-but-finite gradient spike (e.g. from the energy
            # penalty on an unusual batch) before it can meaningfully corrupt
            # the model in one step. 5.0 is a standard default, not tuned to
            # this project specifically.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_nlls.append(nll.item())
            train_penalties.append(penalty.item())

        model.eval()
        with torch.no_grad():
            val_nll = cinn_nll(model, y_val_t, X_val_t)
            val_penalty = energy_conservation_penalty(model, X_val_t, resolved["truth"],
                                                         y_mean_t, y_scale_t, target_masses)
            val_loss = val_nll + ENERGY_PENALTY_WEIGHT * val_penalty
            val_residual = None
            if residual_weight:
                val_residual = residual_agreement_penalty(model, X_val_t, y_val_t)
                val_loss = val_loss + residual_weight * val_residual
                val_residual = val_residual.item()
            elif energy_score_weight:
                val_residual = energy_score_penalty(model, X_val_t, y_val_t)
                val_loss = val_loss + energy_score_weight * val_residual
                val_residual = val_residual.item()
            val_loss = val_loss.item()
            val_nll, val_penalty = val_nll.item(), val_penalty.item()
        train_eval_loss = compute_eval_nll(model, y_train_t, X_train_t, batch_size)

        scheduler.step(val_loss)
        train_loss = float(np.mean(train_losses))
        train_loss_history.append(train_loss)
        train_eval_loss_history.append(train_eval_loss)
        val_loss_history.append(val_loss)
        residual_str = ""
        if residual_weight or energy_score_weight:
            residual_str = (f"  train_residual {np.mean(train_residuals):.4f}  "
                             f"val_residual {val_residual:.4f}")
        print(f"epoch {epoch:4d}  train_nll {np.mean(train_nlls):.4f}  "
              f"train_penalty {np.mean(train_penalties):.6f}  "
              f"train_nll(eval) {train_eval_loss:.4f}  "
              f"val_nll {val_nll:.4f}  val_penalty {val_penalty:.6f}{residual_str}  val_loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(args.output, model, optimizer, scheduler, epoch, val_loss,
                             config, resolved, x_scaler, y_scaler, args.val_fold,
                             residual_weight or energy_score_weight,
                             "mse" if residual_weight else ("energy_score" if energy_score_weight else None))
            print(f"  -> saved checkpoint (val_loss {val_loss:.4f})")
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
                     config, resolved, x_scaler, y_scaler, val_fold,
                     residual_penalty_weight=0.0, residual_penalty_type=None):
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
        "energy_penalty_weight": ENERGY_PENALTY_WEIGHT,
        "residual_penalty_weight": residual_penalty_weight,
        "residual_penalty_type": residual_penalty_type,  # "mse" | "energy_score" | None
    }, path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Hjj unfolding cINN.")
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--preprocessed", required=True, help="Preprocessed HDF5 from preprocessing_training.py")
    p.add_argument("--output", required=True, help="Checkpoint output path (.pt)")
    p.add_argument("--val-fold", type=int, default=4, help="Which fold to hold out for validation")
    p.add_argument("--residual-penalty-weight", type=float, default=0.0,
                    help="Weight on residual_agreement_penalty (raw per-draw MSE to truth, scaled "
                         "feature space). Default 0.0 = off. See the penalty's module docstring for "
                         "the collapse risk -- energy_score_penalty is the documented safer "
                         "alternative, prefer --energy-score-weight instead unless specifically "
                         "testing the raw-MSE formulation.")
    p.add_argument("--energy-score-weight", type=float, default=0.0,
                    help="Weight on energy_score_penalty (proper scoring rule: accuracy vs. truth "
                         "minus a spread term between two independent posterior draws, scaled "
                         "feature space). Default 0.0 = off. Mutually exclusive with "
                         "--residual-penalty-weight -- this is the preferred formulation per "
                         "CLAUDE.md's dphi_jj residual/loss discussion.")
    return p


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()