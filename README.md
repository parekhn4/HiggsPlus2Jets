# HiggsPlus2Jets

Unfolding code for a Hjj analysis. Unfolds Delphes H+2jet events (reco ->
parton level) using a conditional normalizing flow (cINN), to reconstruct obervables for a Higgs-top Yukawa
CP measurement. The model is trained on all three EFT scenarios pooled
together without a CP-scenario label.

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
reco event, the on-shell mean, or both, in one command (`--n-samples 1` is
enough if you only want `--output-draw`; use more, e.g. 500, if you also
want `--output-mean` to be a real average):

```
python scripts/unfold_and_average.py --checkpoint best_model.pt --config configs/no_energy.yaml \
    --output-draw draws.h5 --n-samples 1 --discard-samples
```

`--data-dir` defaults to the config's own `data.input_dir` -- only pass it
to point at data somewhere other than what the config says.

Model/data outputs (checkpoints, preprocessed data, plots) go under
`runs/<date>_<config-name>_<n_blocks>b/`, gitignored.
