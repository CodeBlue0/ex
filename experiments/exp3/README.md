# Experiment 3

## Goal
- Keep the MNIST binary setup from Experiment 1.
- Add per-feature normalization before gate:
  `x_norm = (x - mean_feature) / std_feature`.
- Compute SIM using cosine similarity on raw `g` space.
- Highlight the same SIM group in both `g-space` and `z-space`.

## Current status
- Script is ready to run from this repository without manual path edits.
- Default outputs are saved under `experiments/exp3/` with suffix `_exp3_featnorm_rawg`.

## Run
```bash
python experiments/exp3/mnist_flat_dnn_gate_compare_exp3.py
```

## Main defaults
- data-dir: `/workspace/mnist_data` (aligned with exp1 default)
- device: `CUDA only` (GPU required, no CPU fallback)
- sim-threshold: `0.85` (cosine threshold on raw g-space, `cos >= sim-threshold`)
- min-sim-count: `30` (if SIM count is too small, effective threshold is auto-lowered to include at least this many samples)
- key-topk: `3`
- feature-norm-eps: `1e-6`
