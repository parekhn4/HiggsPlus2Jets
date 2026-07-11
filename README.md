# HiggsPlus2Jets

Unfolding code for a Hjj analysis. Unfolds Delphes H+2jet events (reco ->
parton level) using a conditional normalizing flow (cINN), to reconstruct
CP-sensitive kinematics -- in particular `dphi_jj` -- for a Higgs-top Yukawa
CP measurement. The model is trained on all three EFT scenarios pooled
together, without a CP-scenario label, since real data won't come labeled
either.

## Setup

```
pip install -r requirements.txt
```

## Quickstart

Run the full pipeline (preprocess -> train -> evaluate -> validate) in one command:

```
python scripts/run_pipeline.py --config configs/no_energy.yaml
```

Unfold real/analysis data with a trained checkpoint -- one four-vector per
reco event, the on-shell mean, or both:

```
python inference/inference.py --checkpoint best_model.pt --config configs/no_energy.yaml \
    --data-dir Delphes_Data/ --output samples.h5 --n-samples 1
python scripts/reduce_posterior.py --input samples.h5 --output-draw draws.h5 --output-mean means.h5
```

Model/data outputs (checkpoints, preprocessed data, plots) go under
`runs/<date>_<config-name>_<n_blocks>b/`, gitignored.
