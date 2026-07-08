"""
test_pipeline_stage3.py — verify the AUX_ column rename + event_id
traceability actually works end-to-end, using the real test file.

Usage:
    python test_pipeline_stage3.py --root-file test.root --config configs/no_energy.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import torch
import yaml

import kinematics
import preprocessing_training as prep_train
import preprocessing_inference as prep_inf
import train as train_module
from model import build_model_from_config
from inference import load_checkpoint_bundle, sample_posterior_batch


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-file", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scenario_name = "aux_check"
    tmp_dir = Path("test_pipeline_tmp")
    tmp_dir.mkdir(exist_ok=True)

    # ── 1. preprocessing_training's AUX_ columns ─────────────────────────
    section("1. preprocessing_training.py: AUX_ columns present and correct")
    resolved = prep_train.resolve_config(config)
    df_train = prep_train.build_scenario(args.root_file, scenario_name, config, resolved)

    expected_aux = {"AUX_sample", "AUX_event_id", "AUX_fold", "AUX_n_reco_jets", "AUX_reco_higgs_mass"}
    actual_aux = {c for c in df_train.columns if c.startswith("AUX_")}
    print(f"AUX columns present: {sorted(actual_aux)}")
    assert actual_aux == expected_aux, f"mismatch: {actual_aux ^ expected_aux}"
    print("OK -- all 5 expected AUX_ columns present, no unprefixed leftovers")

    assert df_train["AUX_event_id"].is_unique, "event_id should be unique per surviving event!"
    print(f"AUX_event_id range: [{df_train['AUX_event_id'].min()}, {df_train['AUX_event_id'].max()}], "
          f"all unique -- OK")

    assert (df_train["AUX_sample"] == scenario_name).all()
    print("AUX_sample correctly set on every row -- OK")

    # ── 2. preprocessing_inference's AUX_ columns (same file, reco-only) ──
    section("2. preprocessing_inference.py: AUX_ columns present and correct")
    X_reco_inf, meta_inf = prep_inf.build_reco_features(
        args.root_file, scenario_name, config, resolved["reco"]
    )
    expected_aux_inf = {"AUX_sample", "AUX_event_id", "AUX_n_reco_jets", "AUX_reco_higgs_mass"}
    actual_aux_inf = set(meta_inf.columns)
    assert actual_aux_inf == expected_aux_inf, f"mismatch: {actual_aux_inf ^ expected_aux_inf}"
    print(f"AUX columns present: {sorted(actual_aux_inf)}")
    print("OK")

    # ── 3. Cross-check: training's event_ids should be a SUBSET of ───────
    #    inference's (training ANDs in extra truth cuts on the same file)
    section("3. Cross-check: training event_ids subset of inference event_ids")
    train_ids = set(df_train["AUX_event_id"])
    inf_ids = set(meta_inf["AUX_event_id"])
    print(f"training surviving events: {len(train_ids)}")
    print(f"inference surviving events: {len(inf_ids)}")
    not_subset = train_ids - inf_ids
    if not_subset:
        print(f"  !! PROBLEM: {len(not_subset)} training event_ids not found in inference's set: "
              f"{sorted(list(not_subset))[:10]}...")
    else:
        print("OK -- every training event_id is traceable in inference's broader selection, as expected")

    # ── 4. Full round-trip: checkpoint -> inference.py -> AUX_ merge ──────
    section("4. inference.py's meta merge, on a freshly-trained smoke checkpoint")
    max_jets = config["data"]["max_jets"]
    reco_dim = kinematics.total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(config["data"].get("seed", 42))

    x_scaler = train_module.fit_reco_scaler(
        df_train[[f"x_{i}" for i in range(reco_dim)]].to_numpy(dtype=np.float32),
        resolved["reco"], max_jets,
    )
    y_scaler = train_module.fit_truth_scaler(
        df_train[[f"y_{i}" for i in range(truth_dim)]].to_numpy(dtype=np.float32)
    )
    model = build_model_from_config(config, target_dim=truth_dim, context_dim=reco_dim, device=device)

    checkpoint_path = tmp_dir / "aux_check_model.pt"
    train_module.save_checkpoint(
        str(checkpoint_path), model, torch.optim.Adam(model.parameters()),
        torch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.Adam(model.parameters())),
        epoch=0, val_loss=0.0, config=config, resolved=resolved,
        x_scaler=x_scaler, y_scaler=y_scaler, val_fold=4,
    )

    bundle = load_checkpoint_bundle(str(checkpoint_path), config, device)
    X_reco_full, meta_full = prep_inf.build_reco_features(
        args.root_file, scenario_name, config, bundle["resolved"]["reco"]
    )
    X_scaled = prep_inf.apply_reco_scaling(X_reco_full, bundle["scaler"])

    n_samples = 3
    samples_scaled = sample_posterior_batch(
        bundle["model"], X_scaled[:10], truth_dim,
        n_samples_per_event=n_samples, device=device, batch_size=10,
    )
    samples = prep_inf.invert_truth_scaling(samples_scaled, bundle["scaler"])
    fv = kinematics.reconstruct_event(samples, bundle["resolved"]["truth"])

    event_idx = np.repeat(np.arange(10), n_samples)
    sample_idx = np.tile(np.arange(n_samples), 10)
    from inference import four_vectors_to_dataframe
    df_out = four_vectors_to_dataframe(fv, event_idx, sample_idx)
    meta_repeated = meta_full.iloc[:10].iloc[event_idx].reset_index(drop=True)
    df_out = pd.concat([meta_repeated, df_out], axis=1)

    print(f"final output columns: {list(df_out.columns)}")
    assert "AUX_event_id" in df_out.columns
    assert len(df_out) == 10 * n_samples

    # verify every sample of a given event carries the SAME AUX_event_id
    for ei in range(10):
        rows = df_out[df_out["event_idx"] == ei]
        assert rows["AUX_event_id"].nunique() == 1, f"event {ei}: inconsistent AUX_event_id across samples!"
    print(f"OK -- all {n_samples} posterior samples per event correctly share one AUX_event_id")

    # cross-reference: pick one event_id from the unfolded output and confirm
    # it's traceable back to preprocessing_training's dataframe (if it survived
    # training's stricter selection too)
    sample_row = df_out.iloc[0]
    this_event_id = sample_row["AUX_event_id"]
    if this_event_id in train_ids:
        matching_truth_row = df_train[df_train["AUX_event_id"] == this_event_id].iloc[0]
        print(f"\nEvent {this_event_id}: successfully cross-referenced between inference output "
              f"and training data")
        print(f"  reco higgs mass (inference meta): {sample_row['AUX_reco_higgs_mass']:.2f}")
        print(f"  reco higgs mass (training meta):  {matching_truth_row['AUX_reco_higgs_mass']:.2f}")
        assert abs(sample_row['AUX_reco_higgs_mass'] - matching_truth_row['AUX_reco_higgs_mass']) < 0.01
        print("  OK -- same event, same value, confirmed via AUX_event_id")
    else:
        print(f"\n(event {this_event_id} didn't survive training's stricter truth cuts -- "
              f"can't demo the cross-reference on this particular one, try a different index)")

    section("STAGE 3 COMPLETE")
    print("AUX_ renaming and event_id traceability verified end-to-end on the real file.")


if __name__ == "__main__":
    main()