# Experiment 4

## Goal
- Base experiment: Exp1
- Use two gates instead of one:
  - `g_a = softmax(gate_a_net(x))`
  - `g_b = softmax(gate_b_net(x))`
- Define z-space with two-gate difference:
  - `z = (g_a - g_b) * x`

## Spaces
- `g-space_a`: `g_a`
- `g-space_b`: all samples use `g_a`, but anchor sample uses `g_b[anchor_idx]`
- `z-space`: `(g_a - g_b) * x`

## SIM definitions
- `SIM_a`: cosine similarity on `g_a` to anchor (`cos(g_a, anchor_a) >= threshold`)
- `SIM_b`: cosine similarity between sample `g_a` and anchor `g_b`
  - `cos(g_a[i], g_b[anchor_idx]) >= threshold`

## Outputs
- Pred-label maps:
  - `mnist_ga_space_tsne_pred01_exp4_twogate.png`
  - `mnist_gb_space_tsne_pred01_exp4_twogate.png`
  - `mnist_z_space_tsne_pred01_exp4_twogate.png`
- SIM_a highlighted:
  - `mnist_ga_space_tsne_sim_a_exp4_twogate.png`
  - `mnist_gb_space_tsne_sim_a_exp4_twogate.png`
  - `mnist_z_space_tsne_sim_a_exp4_twogate.png`
- SIM_b highlighted:
  - `mnist_ga_space_tsne_sim_b_exp4_twogate.png`
  - `mnist_gb_space_tsne_sim_b_exp4_twogate.png`
  - `mnist_z_space_tsne_sim_b_exp4_twogate.png`
- Digit 0-9 color maps (base + SIM_a + SIM_b) are also generated with `_digit10_` in filename.
- CSVs:
  - `mnist_gate_sample_metrics_exp4_twogate.csv` (anchor row only)
  - `mnist_anchor_gate_values_exp4_twogate.csv`
  - `mnist_gate_stats_sim_a_exp4_twogate.csv`
  - `mnist_gate_stats_sim_b_exp4_twogate.csv`

## Run
```bash
python experiments/exp4/mnist_flat_dnn_gate_compare_exp4.py
```

## Main defaults
- data-dir: `/workspace/ex/mnist_data` if exists, else `/workspace/mnist_data`
- sim-threshold: `0.85`
- anchor-idx: `892`
