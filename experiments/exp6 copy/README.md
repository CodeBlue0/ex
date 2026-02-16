# Experiment 6

## Goal
- Base experiment: Exp5
- Keep Exp5 feature normalization before gates:
  - `x_norm = (x - mean_feature) / std_feature`
- Two-gate branches:
  - `g_a = softmax(gate_a_net(x_norm))`
  - `g_b = softmax(gate_b_net(x_norm))`
- Branch inputs:
  - `z_a = g_a * x_norm`
  - `z_b = g_b * x_norm`
- Use the same shared backbone for both branches, then take output difference:
  - `logits_a = backbone(z_a)`
  - `logits_b = backbone(z_b)`
  - `logits = logits_a - logits_b`

## Spaces
- `g-space_a`: `g_a`
- `g-space_b`: all samples use `g_a`, but anchor sample uses `g_b[anchor_idx]`
- `g-diff space`: `g_a - g_b`
- `z-space`: `z_a - z_b`

## SIM definitions
- `SIM_a`: `cos(g_a[i], g_a[anchor_idx]) >= threshold`
- `SIM_b`: `cos(g_a[i], g_b[anchor_idx]) >= threshold`

## Outputs
- Same image/CSV set as Exp5, with suffix `_exp6_featnorm_twogate_backbone_diff`.

## Run
```bash
python experiments/exp6/mnist_flat_dnn_gate_compare_exp6.py
```

## Main defaults
- data-dir: `/workspace/ex/mnist_data` if exists, else `/workspace/mnist_data`
- sim-threshold: `0.85`
- anchor-idx: `892`
- feature-norm-eps: `1e-6`
