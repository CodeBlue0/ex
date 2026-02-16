# Experiment 5

## Goal
- Base experiment: Exp4
- Add Exp3-style per-feature normalization before two gates:
  - `x_norm = (x - mean_feature) / std_feature`
- Two-gate model:
  - `g_a = softmax(gate_a_net(x_norm))`
  - `g_b = softmax(gate_b_net(x_norm))`
- z-space:
  - `z = (g_a - g_b) * x_norm`

## Spaces
- `g-space_a`: `g_a`
- `g-space_b`: all samples use `g_a`, but anchor sample uses `g_b[anchor_idx]`
- `g-diff space`: `g_a - g_b`
- `z-space`: `(g_a - g_b) * x_norm`

## SIM definitions
- `SIM_a`: `cos(g_a[i], g_a[anchor_idx]) >= threshold`
- `SIM_b`: `cos(g_a[i], g_b[anchor_idx]) >= threshold`

## Outputs
- Pred-label maps:
  - `mnist_ga_space_tsne_pred01_exp5_featnorm_twogate.png`
  - `mnist_gb_space_tsne_pred01_exp5_featnorm_twogate.png`
  - `mnist_gdiff_space_tsne_pred01_exp5_featnorm_twogate.png`
  - `mnist_z_space_tsne_pred01_exp5_featnorm_twogate.png`
- Digit 0-9 maps:
  - `mnist_ga_space_tsne_digit10_exp5_featnorm_twogate.png`
  - `mnist_gb_space_tsne_digit10_exp5_featnorm_twogate.png`
  - `mnist_gdiff_space_tsne_digit10_exp5_featnorm_twogate.png`
  - `mnist_z_space_tsne_digit10_exp5_featnorm_twogate.png`
- SIM_a highlighted (pred/digit10):
  - `mnist_ga_space_tsne_sim_a_exp5_featnorm_twogate.png`
  - `mnist_gb_space_tsne_sim_a_exp5_featnorm_twogate.png`
  - `mnist_gdiff_space_tsne_sim_a_exp5_featnorm_twogate.png`
  - `mnist_z_space_tsne_sim_a_exp5_featnorm_twogate.png`
  - `mnist_ga_space_tsne_sim_a_digit10_exp5_featnorm_twogate.png`
  - `mnist_gb_space_tsne_sim_a_digit10_exp5_featnorm_twogate.png`
  - `mnist_gdiff_space_tsne_sim_a_digit10_exp5_featnorm_twogate.png`
  - `mnist_z_space_tsne_sim_a_digit10_exp5_featnorm_twogate.png`
- SIM_b highlighted (pred/digit10):
  - `mnist_ga_space_tsne_sim_b_exp5_featnorm_twogate.png`
  - `mnist_gb_space_tsne_sim_b_exp5_featnorm_twogate.png`
  - `mnist_gdiff_space_tsne_sim_b_exp5_featnorm_twogate.png`
  - `mnist_z_space_tsne_sim_b_exp5_featnorm_twogate.png`
  - `mnist_ga_space_tsne_sim_b_digit10_exp5_featnorm_twogate.png`
  - `mnist_gb_space_tsne_sim_b_digit10_exp5_featnorm_twogate.png`
  - `mnist_gdiff_space_tsne_sim_b_digit10_exp5_featnorm_twogate.png`
  - `mnist_z_space_tsne_sim_b_digit10_exp5_featnorm_twogate.png`
- CSVs:
  - `mnist_gate_sample_metrics_exp5_featnorm_twogate.csv` (anchor row only)
  - `mnist_anchor_gate_values_exp5_featnorm_twogate.csv`
  - `mnist_gate_stats_sim_a_exp5_featnorm_twogate.csv`
  - `mnist_gate_stats_sim_b_exp5_featnorm_twogate.csv`

## Run
```bash
python experiments/exp5/mnist_flat_dnn_gate_compare_exp5.py
```

## Main defaults
- data-dir: `/workspace/ex/mnist_data` if exists, else `/workspace/mnist_data`
- sim-threshold: `0.85`
- anchor-idx: `892`
- feature-norm-eps: `1e-6`
