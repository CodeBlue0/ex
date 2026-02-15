# Experiment 2

## Goal
- Keep the same MNIST binary setup as Experiment 1.
- Change SIM computation to use log-centered `g` space:
  `e = log(m + eps) - mean(log(m + eps))`.
- Define SIM with cosine similarity to the anchor in this transformed `g` space.
- Build key-space from anchor top-3 dimensions by default.

## Current status
- Script is ready to run from this repository without manual path edits.
- Default outputs are saved under `experiments/exp2/` with suffix `_exp2_logc`.

## Run
```bash
python experiments/exp2/mnist_flat_dnn_gate_compare_exp2.py
```

## Main defaults
- data-dir: `/workspace/mnist_data` (aligned with exp1 default)
- sim-threshold: `0.998` (cosine threshold for log-centered + L2-normalized g-space)
- min-sim-count: `30` (if threshold is too small, auto-raises to include at least this many SIM samples)
- key-topk: `3`
- log-eps: `1e-8`

같은 라벨 다수 선택됨. 정규화는 그렇게 도움되지는 않는듯.
