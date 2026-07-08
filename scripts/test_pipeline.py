import argparse
import numpy as np
import yaml
import uproot

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# modules from this repo
import catalog
import kinematics
import preprocessing_inference as prep_inf
import preprocessing_training as prep_train


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-file", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── 1. Raw ROOT inspection ──────────────────────────────────────────
    section("1. Raw ROOT file structure")
    with uproot.open(args.root_file) as f:
        print("keys at top level:", f.keys())
        tree = f[config["data"]["tree_name"]]
        branches = tree.keys()
        print(f"tree '{config['data']['tree_name']}' has {len(branches)} branches")

    expected_branches = set(catalog.SCHEMA_MAP[config.get("dataset", "delphes")].values())
    missing = expected_branches - set(branches)
    if missing:
        print(f"  !! MISSING branches expected by catalog.SCHEMA_MAP: {missing}")
    else:
        print(f"  OK -- all {len(expected_branches)} branches catalog.SCHEMA_MAP expects are present")

    # ── 2. Config resolution ────────────────────────────────────────────
    section("2. Config resolution")
    resolved = prep_train.resolve_config(config)
    max_jets = config["data"]["max_jets"]
    reco_dim = kinematics.total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = kinematics.total_dim(resolved["truth"], max_jets=max_jets)
    print(f"reco_dim: {reco_dim}")
    print(f"truth_dim: {truth_dim}")
    print(f"parton_ordering: {resolved['parton_ordering']}")
    print("OK -- config resolved without error")

    # ── 3. Training-side extraction (reco + truth) ──────────────────────
    section("3. preprocessing_training.py: read -> select -> encode")
    native = prep_train.read_native_arrays(args.root_file, config)
    extracted = prep_train.select_and_extract(native, config)
    n_train_events = extracted["n_events"]
    print(f"events surviving FULL selection (reco + truth cuts): {n_train_events}")

    X_reco_train = kinematics.encode_domain(extracted, resolved["reco"], "reco", max_jets=max_jets)
    y_truth = kinematics.encode_domain(extracted, resolved["truth"], "truth", max_jets=max_jets)
    print(f"X_reco shape: {X_reco_train.shape}  (expect ({n_train_events}, {reco_dim}))")
    print(f"y_truth shape: {y_truth.shape}  (expect ({n_train_events}, {truth_dim}))")
    assert X_reco_train.shape == (n_train_events, reco_dim), "X_reco shape mismatch!"
    assert y_truth.shape == (n_train_events, truth_dim), "y_truth shape mismatch!"

    for name, arr in [("X_reco", X_reco_train), ("y_truth", y_truth)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        print(f"{name}: NaN count = {n_nan}, Inf count = {n_inf}"
              + ("  !! PROBLEM" if (n_nan or n_inf) else "  OK"))

    # sanity ranges on raw (pre-encoding) physical quantities
    print(f"\nH_truth pt range: [{extracted['H_truth']['pt'].min():.1f}, {extracted['H_truth']['pt'].max():.1f}] GeV")
    print(f"H_truth eta range: [{extracted['H_truth']['eta'].min():.2f}, {extracted['H_truth']['eta'].max():.2f}]")
    print(f"H_reco mass range: [{extracted['H_reco']['mass'].min():.1f}, {extracted['H_reco']['mass'].max():.1f}] GeV "
          f"(expect within {config['selection']['mass_window_center']} +/- {config['selection']['mass_window_half_width']})")
    print(f"njet range: [{extracted['event_reco']['njet'].min():.0f}, {extracted['event_reco']['njet'].max():.0f}] "
          f"(expect >= 2, <= {max_jets})")
    print(f"dphi_jj range: [{extracted['event_truth']['dphi_jj'].min():.2f}, {extracted['event_truth']['dphi_jj'].max():.2f}] "
          f"(expect within [-pi, pi])")

    # ── 4. Inference-side extraction (reco only, same file) ─────────────
    section("4. preprocessing_inference.py: read -> select -> encode (reco only)")
    X_reco_inf, meta_inf = prep_inf.build_reco_features(
        args.root_file, "test_scenario", config, resolved["reco"]
    )
    n_inf_events = len(X_reco_inf)
    print(f"events surviving reco-ONLY selection: {n_inf_events}")
    print(f"(training-side had {n_train_events} -- inference count should be >= training count, "
          f"since training ANDs in extra truth cuts)")
    assert n_inf_events >= n_train_events, "inference selection is somehow STRICTER than training -- investigate!"

    print(f"\nreco feature statistics comparison (should be similar distributions, "
          f"not identical row-for-row since the event sets differ):")
    print(f"  training X_reco  mean: {X_reco_train.mean():.4f}  std: {X_reco_train.std():.4f}")
    print(f"  inference X_reco mean: {X_reco_inf.mean():.4f}  std: {X_reco_inf.std():.4f}")
    mean_diff = abs(X_reco_train.mean() - X_reco_inf.mean())
    print(f"  mean difference: {mean_diff:.4f} " + ("OK" if mean_diff < 0.5 else "!! LARGE -- check for a real divergence between the two paths"))

    # ── 5. Encode/decode round-trip on THIS file's real data ────────────
    section("5. Round-trip: encode_domain -> decode_domain")
    decoded_reco = kinematics.decode_domain(X_reco_train, resolved["reco"], "reco", max_jets=max_jets)
    orig_H_pt = extracted["H_reco"]["pt"]
    decoded_H_pt = decoded_reco["H_reco"]["pt"]
    max_err = np.max(np.abs(orig_H_pt - decoded_H_pt))
    print(f"H reco pt: max round-trip error = {max_err:.6f} GeV "
          + ("OK" if max_err < 1e-2 else "!! ROUND-TRIP BROKEN"))

    # ── 6. Physics sanity: reconstruct_event on real truth data ─────────
    section("6. kinematics.reconstruct_event on real truth samples")
    fv = kinematics.reconstruct_event(y_truth, resolved["truth"])
    for obj_name in ("H", "j1", "j2"):
        E, px, py, pz = fv[obj_name][:, 0], fv[obj_name][:, 1], fv[obj_name][:, 2], fv[obj_name][:, 3]
        mass = np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0))
        pt = np.sqrt(px**2 + py**2)
        print(f"{obj_name}: pt range [{pt.min():.1f}, {pt.max():.1f}] GeV, "
              f"reconstructed mass mean = {mass.mean():.2f} GeV "
              f"(expect ~{resolved['truth']['objects'][obj_name].get('fixed_mass', 'learned')})")
        if np.any(np.isnan(E)) or np.any(pt <= 0):
            print(f"  !! PROBLEM: NaN energies or non-positive pt in {obj_name}")
        else:
            print(f"  OK")

    # ── 7. End-to-end shape check against the model ──────────────────────
    section("7. Model can actually consume this data's dimensions")
    import torch
    from model import build_model_from_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_from_config(config, target_dim=truth_dim, context_dim=reco_dim, device=device)

    xb = torch.tensor(X_reco_train[:8], dtype=torch.float32, device=device)
    yb = torch.tensor(y_truth[:8], dtype=torch.float32, device=device)
    z, log_det = model(yb, xb)
    print(f"forward pass output: z shape {tuple(z.shape)}, log_det shape {tuple(log_det.shape)}")
    assert z.shape == (8, truth_dim), "model output dim doesn't match truth_dim!"
    print("OK -- model architecture is dimensionally compatible with this file's resolved config")

    z_sample = torch.randn(8, truth_dim, device=device)
    x_inv = model.inverse(z_sample, xb)
    print(f"inverse pass output shape: {tuple(x_inv.shape)}")
    assert x_inv.shape == (8, truth_dim)
    print("OK -- forward and inverse both run without shape errors")

    section("ALL SECTIONS COMPLETE")
    print("If every section above says OK, the pipeline is consistent end-to-end")
    print("against this real file, from raw ROOT branches through to a model")
    print("forward/inverse pass. Ready for an actual training run.")


if __name__ == "__main__":
    main()