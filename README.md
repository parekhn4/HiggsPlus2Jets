# HiggsPlus2Jets
for maintaining unfolding code for a Hjj analysis 

This codebase unfolds Delphes H+2jet events (reco -> parton level) using a conditional
normalizing flow (cINN). Given detector-level info (the Higgs from the two
photons, and the jets), it samples the posterior over the true H, j1, j2
kinematics, including Delta_phi_jj, which is what we actually care about for
the CP measurement.

The model doesn't know which EFT scenario (at_1_bt_0, at_0_bt_1, at_1_bt_1)
an event came from. It's trained on all three pooled together, on purpose --
real data won't come labeled by CP scenario either, so the model has to be
able to unfold without that info.

## Files

- `catalog.py` - list of physical variables you can train on (pt, eta, phi,
  mass, E, dphi_jj, njet, ...) and how each one gets encoded (log1p, sin/cos,
  etc). If you want to add a new variable, this is where it goes.
- `kinematics.py` - the actual math. delta_phi, four-vector building,
  encode/decode between physical values and network features. 
- `preprocessing_inference.py` - ROOT -> reco features only. This is what
  one would run on real data or any file that doesn't have truth info.
- `preprocessing_training.py` - ROOT -> reco + truth features, plus a fold
  column for train/val splitting. Needs the Particle.* branches, so this
  only works on simulated events. Reuses preprocessing_inference for the
  reco half instead of redoing that work.
- `model.py` - the cINN itself (RQS coupling blocks). Pure architecture, no
  config file reading beyond getting told the dims.
- `train.py` - trains the model, saves a checkpoint that has everything in
  it (weights, scaler, which config built it, which h5 file it trained on).
- `inference.py` - loads a checkpoint, samples the posterior, writes every
  sampled four-vector per event to HDF5 (nothing averaged).
- `reduce_posterior.py` - collapses `inference.py`'s output to one on-shell
  four-vector per event (mean pt/eta/phi, fixed mass, energy recomputed).
- `unfold_and_average.py` - runs the two above back to back in one command.
- `evaluate.py` - like inference but on the held-out validation fold, so you
  can actually check if it's working (closure plots comparing truth vs reco
  vs unfolded).
- `plotting.py` - the actual plot-drawing code, used by evaluate.py.

## The scalers 

The reco/truth features get standardized (mean 0, std 1) before hitting the
network. This is fit ONCE on the training data and then reused everywhere
else -- val set, inference, whatever. Never refit on new data, because the
network's weights are tied to whatever numbers it was trained on.

Jets are a bit special: since most events don't have all 12 jet slots
filled, we only fit the scaler on jet slot 1 (which is always
there) and copy those same numbers to slots 2-12. If you check the scaled
data and slot 12's mean looks way off from 0, that's expected -- it's mostly
zero-padding getting scaled by jet-1 stats and not a bug.

All of this scaler info gets saved directly into the checkpoint now.

## Delta_phi_jj

- Training uses whatever `parton_ordering` the config says (pt or eta) to
  decide which parton is "j1" and pair them up. Doesn't matter much which
  one, it's just a consistent labeling for the network to learn.
- The actual CP-sensitive angle needs jets ordered by SIGNED eta (forward
  jet first). Thre is a risk of washing out CP asymmetry with wrong ordering. 
  `kinematics.eta_ordered_dphi_jj` does this correctly, and it's computed
  fresh from the four-vectors regardless of what ordering training used.
  `evaluate.py` reports both versions so you can tell them apart.

## How to run 

Set up the environment (need torch, uproot, awkward, pandas, sklearn, yaml,
tables, matplotlib). One can create a venv and inside it:

```
pip install -r requirements.txt
```

1. Preprocess (needs the full Delphes files with truth branches):

```
python preprocessing_training.py --config configs/no_energy.yaml --output preprocessed.h5
```

2. Train:

```
python train.py --config configs/no_energy.yaml --preprocessed preprocessed.h5 --output best_model.pt --val-fold 4
```

`--val-fold` picks which of the 5 folds gets held out for validation. Folds
are assigned once during preprocessing and don't change between runs.

3. Unfold (only needs Photon/Jet branches, works on real data too). Get the
   full posterior, the per-event mean, or both in one command:

```
# full posterior: samples.h5[scenario]/four_vectors/{H,j1,j2}, shape (n_events, n_samples, 4)
python inference.py --checkpoint best_model.pt --config configs/no_energy.yaml \
    --data-dir Delphes_Data/ --output samples.h5 --n-samples 500 [--batch-size 512] [--seed 42]

# collapse to one on-shell four-vector/event
python reduce_posterior.py --input samples.h5 --output means.h5

# or both at once (keeps the full-sample file by default; --discard-samples to drop it,
# --samples-output PATH to control where it's kept)
python unfold_and_average.py --checkpoint best_model.pt --config configs/no_energy.yaml \
    --data-dir Delphes_Data/ --output means.h5 --n-samples 500
```

4. Check that it actually works (closure plots on the held-out fold):

```
python evaluate.py --checkpoint best_model.pt --config configs/no_energy.yaml --preprocessed preprocessed.h5 --n-samples 200 --output-dir eval_plots/
```

This makes one plot per scenario plus one pooled plot (all scenarios
combined, no CP label) since that pooled one is the closer match to what
evaluating on real data will actually look like.

## Adding a new variable to train on

Add it to `catalog.CATALOG` in catalog.py (a few lines - what it's called,
what object(s) it can go on, default encoding). Then just add it to a
config's `variables:` list for whichever object needs it. Nothing else
needs to change - dims get recalculated automatically.

## If you want to try training on Higgs/jet energy instead of fixed masses

Copy configs/no_energy.yaml to something like configs/with_energy.yaml,
change the variable lists (add E to H, add mass to j1/j2 instead of fixing
it).
