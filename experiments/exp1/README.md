# Experiment 1

## Setup
- Task: MNIST binary classification (0-4 vs 5-9)
- Model: `GatedDNN` with softmax gate
- Device: CUDA
- Epochs: 6
- Batch size: 512
- SIM threshold: 0.20

## Key outputs
- `mnist_g_space_tsne_pred01.png`
- `mnist_z_space_tsne_pred01.png`
- `mnist_g_space_tsne_sim_from_g.png`
- `mnist_z_space_tsne_sim_from_g.png`
- `mnist_key_space_tsne_pred01.png`
- `mnist_key_space_tsne_dims_542_157.png`

## Last observed metrics
- baseline_test_acc: 0.9818
- gated_test_acc: 0.9788
- delta_test_acc: -0.0030
- sim_anchor_idx: 267
- sim_ratio (@0.20): 0.072333
- fixed key dims: [542, 157]



너무 노이즈가 많고 같은 label만 선택됨