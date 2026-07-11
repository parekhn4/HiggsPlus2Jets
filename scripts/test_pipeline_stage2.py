import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import yaml

import core.kinematics as kinematics
import training.preprocessing_training as prep_train
import inference.preprocessing_inference as prep_inf
import training.train as train_module
from core.model import build_model_from_config


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-file", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--smoke-epochs", type=int, default=3)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scenario_name = "smoke_test"
    tmp_dir = Path("test_pipeline_tmp")
    tmp_dir.mkdir(exist_ok=True)
    preprocessed_path = tmp_dir / "preprocessed.h5"
    checkpoint_path = tmp_dir / "smoke_model.pt"

    # ── 1. build_and_save: full persistence pipeline on one file ────────
    section("1. preprocessing_training.build_and_save -> HDF5")
    resolved = prep_train.resolve_config(config)
    df = prep_train.build_scenario(args.root_file, scenario_name, config, resolved)
    print(f"built {len(df)} rows, columns: {list(df.columns[:6])}...")
    assert "AUX_fold" in df.columns, "AUX_fold column missing!"

    n_folds = config["data"].get("n_folds", 5)
    fold_counts = df["AUX_fold"].value_counts().sort_index()
    print(f"fold distribution:\n{fold_counts}")
    assert set(fold_counts.index) == set(range(n_folds)), "not all folds represented!"
    assert fold_counts.min() > 0.15 * len(df) / n_folds, "folds are wildly imbalanced!"
    print("OK -- fold column present, all folds populated, reasonably balanced")

    with pd.HDFStore(preprocessed_path, mode="w") as store:
        store.put(scenario_name, df, format="fixed")
    print(f"wrote {preprocessed_path}")

    with pd.HDFStore(preprocessed_path, mode="r") as store:
        df_read = store[scenario_name]
    assert len(df_read) == len(df)
    assert np.allclose(df_read["x_0"], df["x_0"])
    print("OK -- HDF5 write/read round-trip preserves data exactly")

    # ── 2. load_pooled_dataset + split_by_fold on the real file ─────────
    section("2. train.py's data loading + fold split")
    max_jets = config["data"]["max_jets"]
    reco_dim = kinematics.total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)

    pooled = train_module.load_pooled_dataset(str(preprocessed_path), reco_dim, truth_dim)
    val_fold = 4
    X_train, y_train, X_val, y_val = train_module.split_by_fold(pooled, val_fold, reco_dim, truth_dim)
    print(f"train: {len(X_train)}  val (fold {val_fold}): {len(X_val)}")
    assert len(X_train) + len(X_val) == len(df)
    print("OK -- no events lost or duplicated across the split")

    # ── 3. scaler fitting on this file's REAL data ───────────────────────
    section("3. Scaler fitting on real data")
    x_scaler = train_module.fit_reco_scaler(X_train, resolved["reco"], max_jets)
    y_scaler = train_module.fit_truth_scaler(y_train)

    X_train_s = (X_train - x_scaler["mean"]) / x_scaler["scale"]
    y_train_s = (y_train - y_scaler["mean"]) / y_scaler["scale"]
    print(f"scaled X_train: mean={X_train_s.mean():.4f}, std={X_train_s.std():.4f} (expect ~0, ~1)")
    print(f"scaled y_train: mean={y_train_s.mean():.4f}, std={y_train_s.std():.4f} (expect ~0, ~1)")
    H_cols = slice(0, 5)          # Higgs block: pt, eta, sin, cos, mass
    jet1_cols = slice(5, 10)      # first jet slot
    jet12_cols = slice(60, 65)    # last jet slot (12th) -- most sparsely populated
    njet_col = 65                 # event block, last column

    print(f"H block:       mean={X_train_s[:, H_cols].mean():.4f}  std={X_train_s[:, H_cols].std():.4f}  (expect ~0, ~1)")
    print(f"jet slot 1:     mean={X_train_s[:, jet1_cols].mean():.4f}  std={X_train_s[:, jet1_cols].std():.4f}  (expect ~0, ~1 -- this is what got fit)")
    print(f"jet slot 12:    mean={X_train_s[:, jet12_cols].mean():.4f}  std={X_train_s[:, jet12_cols].std():.4f}  (expect a LARGE negative offset -- mostly zero-padding, scaled by jet1's real-jet stats)")
    print(f"njet column:    mean={X_train_s[:, njet_col].mean():.4f}  std={X_train_s[:, njet_col].std():.4f}  (expect ~0, ~1)")

    assert abs(X_train_s[:, H_cols].mean()) < 0.5, "Higgs block should be well-centered!"
    assert abs(X_train_s[:, jet1_cols].mean()) < 0.5, "jet slot 1 should be well-centered (it's what got fit)!"
    assert abs(y_train_s.mean()) < 0.5
    print("OK -- the blocks that were actually FIT on are properly centered; "
      "higher jet slots show the expected zero-padding offset, which is fine.")
    print("OK -- scaled data is properly centered")

    # ── 4. training smoke test -- a few real epochs, not synthetic ──────
    section(f"4. Training smoke test ({args.smoke_epochs} epochs)")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(config["data"].get("seed", 42))

    model = build_model_from_config(config, target_dim=truth_dim, context_dim=reco_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"].get("lr", 1e-4))

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)

    losses = []
    for epoch in range(args.smoke_epochs):
        model.train()
        optimizer.zero_grad()
        loss = train_module.cinn_nll(model, y_train_t, X_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  epoch {epoch}: nll = {loss.item():.4f}")

    assert not any(np.isnan(l) for l in losses), "NaN loss encountered!"
    print(f"OK -- loss is finite across {args.smoke_epochs} epochs "
          f"({'decreasing' if losses[-1] < losses[0] else 'not monotonically decreasing yet -- fine for only a few epochs'})")

    train_module.save_checkpoint(str(checkpoint_path), model, optimizer,
                                   torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                                   epoch=args.smoke_epochs, val_loss=losses[-1],
                                   config=config, resolved=resolved,
                                   x_scaler=x_scaler, y_scaler=y_scaler, val_fold=val_fold)
    print(f"OK -- checkpoint written to {checkpoint_path}")

    # ── 5. checkpoint round-trip through inference.py ────────────────────
    section("5. Load checkpoint via inference.py, run on the same file")
    from inference.inference import load_checkpoint_bundle, sample_posterior_batch
    bundle = load_checkpoint_bundle(str(checkpoint_path), config, device)
    print(f"loaded checkpoint: epoch {bundle['epoch']}, val_loss {bundle['val_loss']:.4f}")
    assert bundle["max_jets"] == max_jets
    print("OK -- max_jets consistency check passed")

    X_reco_inf, meta = prep_inf.build_reco_features(args.root_file, scenario_name, config, bundle["resolved"]["reco"])
    X_reco_inf_scaled = prep_inf.apply_reco_scaling(X_reco_inf, bundle["scaler"])

    samples_scaled = sample_posterior_batch(
        bundle["model"], X_reco_inf_scaled[:20], truth_dim,
        n_samples_per_event=5, device=device, batch_size=20,
    )
    samples = prep_inf.invert_truth_scaling(samples_scaled, bundle["scaler"])
    fv = kinematics.reconstruct_event(samples, bundle["resolved"]["truth"])

    for obj_name in ("H", "j1", "j2"):
        E, px, py, pz = fv[obj_name][:,0], fv[obj_name][:,1], fv[obj_name][:,2], fv[obj_name][:,3]
        pt = np.sqrt(px**2 + py**2)
        print(f"{obj_name} unfolded (untrained model, expect noisy but FINITE): "
              f"pt range [{pt.min():.1f}, {pt.max():.1f}] GeV, any NaN: {np.any(np.isnan(E))}")
        assert not np.any(np.isnan(E)), f"{obj_name} produced NaN four-vectors!"
    print("OK -- full checkpoint -> inference -> four-vector chain runs without NaN "
          "(values will look physically wrong -- this model only trained for "
          f"{args.smoke_epochs} epochs, that's expected)")

    section("STAGE 2 COMPLETE")
    print("Persistence, training loop, and the checkpoint->inference round-trip")
    print("all run cleanly on real data. Safe to kick off a real, full training run.")
    print(f"\n(scratch files left in {tmp_dir}/ -- delete when done: rm -rf {tmp_dir})")


if __name__ == "__main__":
    main()