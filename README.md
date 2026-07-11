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

Grouped by folder, in pipeline order:

- `core/` - shared library code, no CLI of its own.
  - `catalog.py` - list of physical variables you can train on (pt, eta, phi,
    mass, E, dphi_jj, njet, ...) and how each one gets encoded (log1p, sin/cos,
    etc). If you want to add a new variable, this is where it goes.
  - `kinematics.py` - the actual math. delta_phi, four-vector building,
    encode/decode between physical values and network features, config-agnostic
    observable derivation (`build_observables`, `reco_four_vectors`).
  - `model.py` - the cINN itself (RQS coupling blocks). Pure architecture, no
    config file reading beyond getting told the dims.
- `training/`
  - `preprocessing_training.py` - ROOT -> reco + truth features, plus a fold
    column for train/val splitting. Needs the Particle.* branches, so this
    only works on simulated events. Reuses `inference/preprocessing_inference.py`
    for the reco half instead of redoing that work.
  - `train.py` - trains the model, saves a checkpoint that has everything in
    it (weights, scaler, which config built it, which h5 file it trained on),
    plus a `<checkpoint_stem>_loss_curve.png`.
- `inference/`
  - `preprocessing_inference.py` - ROOT -> reco features only. This is what
    one would run on real data or any file that doesn't have truth info.
  - `inference.py` - loads a checkpoint, samples the posterior, writes every
    sampled four-vector per event to HDF5 (nothing averaged).
- `evaluate/evaluate.py` - like inference but on the held-out validation fold,
  so you can actually check if it's working (closure plots comparing truth vs
  reco vs unfolded).
- `plotting/plotting.py` - the actual plot-drawing code, used by `evaluate.py`
  and `validate_unfolding.py`.
- `scripts/` - things you actually run from the CLI, beyond the above.
  - `run_pipeline.py` - runs preprocess -> train -> evaluate -> validate_unfolding
    end to end, writing everything into one auto-named
    `runs/<date>_<config-name>_<n_blocks>b/` folder. The quickest way to kick
    off a full run; see "How to run" below for the individual commands it wraps.
  - `reduce_posterior.py` - collapses `inference.py`'s output to one
    four-vector per event, either as the on-shell mean (avg pt/eta/phi, fixed
    mass, energy recomputed) or a single random posterior draw (already
    unweighted, no re-derivation needed) -- pick one or both.
  - `unfold_and_average.py` - runs `inference.py` + `reduce_posterior.py` back
    to back in one command.
  - `validate_unfolding.py` - on the held-out fold, compares mean vs. single
    draw vs. the full posterior as ways to collapse the samples: closure
    plots for each, plus (truth - reco/mean/draw/samples) residual histograms
    overlaid so you can see which reduction is actually tighter. The mean can
    distort a genuinely multimodal observable (e.g. dphi_jj under a
    jet-labeling ambiguity) in a way a single draw doesn't.
  - `test_pipeline.py`, `test_pipeline_stage2.py` - end-to-end sanity checks
    against a real ROOT file (branch presence, encode/decode round-trip, model
    forward/inverse pass, a few real training epochs). Run these after
    touching `catalog.py`/`kinematics.py`/the config schema, before kicking off
    a real training run.
  - `visualize_model.py` - renders a `torchview` diagram of a checkpoint's
    architecture (self-contained, rebuilds the model from the checkpoint's own
    config -- no separate config file needed). `--depth 1` (default) is the only
    practical setting for a real model; deeper unfolds each block's internal
    tensor ops and blows up in size (a depth-unlimited render of a 24-block
    model produced a 978-million-pixel unusable image). Needs `torchview` +
    graphviz's `dot` binary on PATH.

Model/data outputs (checkpoints, `preprocessed.h5`, plots, unfolded h5 files)
go under `runs/<date>_<config-name>_<n_blocks>b[_suffix]/` -- gitignored,
never committed. The folder name is just a mnemonic; checkpoints carry their
own resolved config/epoch/seed, and `preprocessed.h5` always gets a
`.config.yaml` snapshot next to it, so real provenance never depends on the
folder name being complete.

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

All commands below assume you're running from the repo root. Output paths in
these examples are shown as bare filenames for brevity, but in practice they
should point into a `runs/<date>_<config-name>_<n_blocks>b/` folder (see
"Files" above) -- e.g. `runs/2026-07-10_no_energy_16b/preprocessed.h5`.

**Quickest path for a full training run:** steps 1-4 below (preprocess,
train, evaluate, validate_unfolding) in one command, auto-named into
`runs/`:
```
python scripts/run_pipeline.py --config configs/no_energy.yaml
```
The individual steps, if you want to run them separately or need real
analysis data unfolded (step 3'):

1. Preprocess (needs the full Delphes files with truth branches):

```
python training/preprocessing_training.py --config configs/no_energy.yaml --output preprocessed.h5
```

2. Train:

```
python training/train.py --config configs/no_energy.yaml --preprocessed preprocessed.h5 --output best_model.pt --val-fold 4
```

`--val-fold` picks which of the 5 folds gets held out for validation. Folds
are assigned once during preprocessing and don't change between runs.

3. Unfold (only needs Photon/Jet branches, works on real data too). Get the
   full posterior, a per-event reduction, or both in one command:

```
# full posterior: samples.h5[scenario]/four_vectors/unfolded/{H,j1,j2}, shape (n_events, n_samples, 4)
python inference/inference.py --checkpoint best_model.pt --config configs/no_energy.yaml \
    --data-dir Delphes_Data/ --output samples.h5 --n-samples 1 [--batch-size 512] [--seed 42]

# collapse to one four-vector/event -- give one or both output paths
python scripts/reduce_posterior.py --input samples.h5 \
    --output-mean means.h5 --output-draw draws.h5 [--draw-index 0]

# or both (sampling + reduction) at once (keeps the full-sample file by default;
# --discard-samples to drop it, --samples-output PATH to control where it's kept)
python scripts/unfold_and_average.py --checkpoint best_model.pt --config configs/no_energy.yaml \
    --data-dir Delphes_Data/ --output-mean means.h5 --output-draw draws.h5 --n-samples 1
```

**Use `--n-samples 1` unless you also need `--output-mean` or the full
posterior itself.** A single draw is already an unweighted, exact sample
(no reweighting needed), and unlike the mean it provably preserves the
correct population-level shape when pooled across events -- see
`validate_unfolding.py` below and `CLAUDE.md` for the full reasoning and
literature citation. Sampling more than once per event only pays off if
you're deriving the mean or keeping the full posterior from the same run.

4. Check that it actually works (closure plots on the held-out fold):

```
python evaluate/evaluate.py --checkpoint best_model.pt --config configs/no_energy.yaml --preprocessed preprocessed.h5 --n-samples 200 --output-dir eval_plots/
```

This makes one plot per scenario plus one pooled plot (all scenarios
combined, no CP label) since that pooled one is the closer match to what
evaluating on real data will actually look like.

5. Compare mean vs. single-draw vs. full-posterior reduction against truth (held-out fold):

```
python scripts/validate_unfolding.py --checkpoint best_model.pt --config configs/no_energy.yaml \
    --preprocessed preprocessed.h5 --n-samples 200 --output-dir validation_plots/
```

## Adding a new variable to train on

Add it to `catalog.CATALOG` in `core/catalog.py` (a few lines - what it's called,
what object(s) it can go on, default encoding). Then just add it to a
config's `variables:` list for whichever object needs it. Nothing else
needs to change - dims get recalculated automatically.

## If you want to try training on Higgs/jet energy instead of fixed masses

Copy configs/no_energy.yaml to something like configs/with_energy.yaml,
change the variable lists (add E to H, add mass to j1/j2 instead of fixing
it).
