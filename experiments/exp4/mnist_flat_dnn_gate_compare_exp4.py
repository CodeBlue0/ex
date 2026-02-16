from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TwoGateDNN(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden_dims: tuple[int, ...] = (512, 256, 128), gate_hidden: int = 256, n_classes: int = 2):
        super().__init__()
        self.gate_a_net = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, in_dim),
        )
        self.gate_b_net = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, in_dim),
        )
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], n_classes),
        )

    def forward_with_spaces(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        g_a = torch.softmax(self.gate_a_net(x), dim=1)
        g_b = torch.softmax(self.gate_b_net(x), dim=1)
        z = (g_a - g_b) * x
        logits = self.backbone(z)
        return logits, g_a, g_b, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _, _ = self.forward_with_spaces(x)
        return logits


@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, y_digit in loader:
        xb = xb.to(device)
        y_bin = (y_digit >= 5).long().to(device)
        logits = model(xb)
        loss = criterion(logits, y_bin)
        total_loss += float(loss.item()) * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y_bin).sum().item())
        total += int(xb.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def train_and_eval(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, epochs: int, lr: float) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, y_digit in train_loader:
            xb = xb.to(device)
            y_bin = (y_digit >= 5).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, y_bin)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            seen += int(xb.size(0))

        train_loss = running_loss / max(seen, 1)
        val_loss, val_acc = evaluate_binary(model, val_loader, device)
        print(f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate_binary(model, test_loader, device)
    return {
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "model": model,
    }


@torch.no_grad()
def collect_spaces_from_loader(
    model: TwoGateDNN,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ga_list: list[np.ndarray] = []
    gb_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    digit_list: list[np.ndarray] = []
    collected = 0

    for xb, y_digit in loader:
        xb = xb.to(device)
        logits, g_a, g_b, z = model.forward_with_spaces(xb)
        ga_np = g_a.detach().cpu().numpy()
        gb_np = g_b.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        p_np = logits.argmax(dim=1).detach().cpu().numpy()
        d_np = y_digit.detach().cpu().numpy()

        if collected + ga_np.shape[0] > max_samples:
            keep = max_samples - collected
            if keep <= 0:
                break
            ga_np = ga_np[:keep]
            gb_np = gb_np[:keep]
            z_np = z_np[:keep]
            p_np = p_np[:keep]
            d_np = d_np[:keep]

        ga_list.append(ga_np)
        gb_list.append(gb_np)
        z_list.append(z_np)
        pred_list.append(p_np)
        digit_list.append(d_np)
        collected += ga_np.shape[0]
        if collected >= max_samples:
            break

    return (
        np.concatenate(ga_list, axis=0),
        np.concatenate(gb_list, axis=0),
        np.concatenate(z_list, axis=0),
        np.concatenate(pred_list, axis=0),
        np.concatenate(digit_list, axis=0),
    )


@torch.no_grad()
def save_space_tsne_by_pred(space: np.ndarray, pred_label: np.ndarray, out_path: str, title: str) -> None:
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    emb = tsne.fit_transform(space)
    cmap = np.array(["#1f77b4", "#d62728"])
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=cmap[pred_label], s=8, alpha=0.70)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved_tsne={out_path}")


def save_space_tsne_with_sim(
    space: np.ndarray,
    sim_mask: np.ndarray,
    anchor_idx: int,
    label: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    emb = tsne.fit_transform(space)

    cmap = np.array(["#1f77b4", "#d62728"])
    non = ~sim_mask

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[non, 0], emb[non, 1], c=cmap[label[non]], s=7, alpha=0.10, label="non-SIM")
    plt.scatter(emb[sim_mask, 0], emb[sim_mask, 1], c=cmap[label[sim_mask]], s=11, alpha=0.90, label="SIM")
    plt.scatter(emb[anchor_idx, 0], emb[anchor_idx, 1], c="#facc15", s=110, marker="*", label="anchor")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved_tsne={out_path}")


def save_space_tsne_by_digit10(space: np.ndarray, digit_label: np.ndarray, out_path: str, title: str) -> None:
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    emb = tsne.fit_transform(space)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=cmap(np.mod(digit_label, 10)), s=8, alpha=0.70)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    handles = [Line2D([], [], marker="o", linestyle="", color=cmap(i), markersize=6, label=str(i)) for i in range(10)]
    plt.legend(handles=handles, loc="best", title="Digit", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved_tsne={out_path}")


def save_space_tsne_with_sim_digit10(
    space: np.ndarray,
    sim_mask: np.ndarray,
    anchor_idx: int,
    digit_label: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    emb = tsne.fit_transform(space)
    cmap = plt.get_cmap("tab10")
    non = ~sim_mask

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[non, 0], emb[non, 1], c=cmap(np.mod(digit_label[non], 10)), s=7, alpha=0.10)
    plt.scatter(emb[sim_mask, 0], emb[sim_mask, 1], c=cmap(np.mod(digit_label[sim_mask], 10)), s=11, alpha=0.90)
    plt.scatter(emb[anchor_idx, 0], emb[anchor_idx, 1], c="#facc15", s=110, marker="*")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    status_handles = [
        Line2D([], [], marker="o", linestyle="", color="#666666", alpha=0.2, markersize=6, label="non-SIM"),
        Line2D([], [], marker="o", linestyle="", color="#111111", alpha=0.9, markersize=6, label="SIM"),
        Line2D([], [], marker="*", linestyle="", color="#facc15", markersize=12, label="anchor"),
    ]
    leg1 = plt.legend(handles=status_handles, loc="upper right", title="Group")
    plt.gca().add_artist(leg1)

    digit_handles = [Line2D([], [], marker="o", linestyle="", color=cmap(i), markersize=6, label=str(i)) for i in range(10)]
    plt.legend(handles=digit_handles, loc="lower right", title="Digit", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved_tsne={out_path}")


def cosine_similarity_to_anchor(space: np.ndarray, anchor_idx: int) -> np.ndarray:
    normed = space / np.maximum(np.linalg.norm(space, axis=1, keepdims=True), 1e-12)
    anchor = normed[anchor_idx]
    return normed @ anchor


def cosine_similarity_to_anchor_vector(space: np.ndarray, anchor_vec: np.ndarray) -> np.ndarray:
    normed = space / np.maximum(np.linalg.norm(space, axis=1, keepdims=True), 1e-12)
    anchor_normed = anchor_vec / max(float(np.linalg.norm(anchor_vec)), 1e-12)
    return normed @ anchor_normed


def build_sim_mask(space: np.ndarray, anchor_idx: int, threshold: float) -> tuple[np.ndarray, float, bool]:
    c = cosine_similarity_to_anchor(space, anchor_idx)
    sim_mask = c >= threshold
    return sim_mask, threshold, False


def build_sim_mask_from_anchor_vector(
    space: np.ndarray,
    anchor_vec: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float, bool]:
    c = cosine_similarity_to_anchor_vector(space, anchor_vec)
    sim_mask = c >= threshold
    return sim_mask, threshold, False


def _entropy_rows(x: np.ndarray) -> np.ndarray:
    x_safe = np.clip(x, 1e-12, 1.0)
    return -np.sum(x_safe * np.log(x_safe), axis=1)


def _topk_sum_rows(x: np.ndarray, k: int) -> np.ndarray:
    kk = max(1, min(int(k), x.shape[1]))
    part = np.partition(x, x.shape[1] - kk, axis=1)[:, -kk:]
    return np.sum(part, axis=1)


def save_gate_csvs(
    g_a: np.ndarray,
    g_b: np.ndarray,
    pred_label: np.ndarray,
    true_digit: np.ndarray,
    anchor_idx: int,
    sim_a: np.ndarray,
    sim_b: np.ndarray,
    sample_metrics_csv_out: str,
    anchor_gate_csv_out: str,
    sim_a_gate_stats_csv_out: str,
    sim_b_gate_stats_csv_out: str,
) -> None:
    _, d = g_a.shape

    ga_max = np.max(g_a, axis=1)
    gb_max = np.max(g_b, axis=1)
    ga_top3 = _topk_sum_rows(g_a, 3)
    gb_top3 = _topk_sum_rows(g_b, 3)
    ga_entropy = _entropy_rows(g_a)
    gb_entropy = _entropy_rows(g_b)

    with open(sample_metrics_csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample_pos",
                "true_digit_label",
                "pred_label_bin",
                "is_anchor",
                "in_sim_a",
                "in_sim_b",
                "ga_max",
                "ga_top3_sum",
                "ga_entropy",
                "gb_max",
                "gb_top3_sum",
                "gb_entropy",
            ]
        )
        i = anchor_idx
        w.writerow(
            [
                int(i),
                int(true_digit[i]),
                int(pred_label[i]),
                1,
                int(sim_a[i]),
                int(sim_b[i]),
                float(ga_max[i]),
                float(ga_top3[i]),
                float(ga_entropy[i]),
                float(gb_max[i]),
                float(gb_top3[i]),
                float(gb_entropy[i]),
            ]
        )
    print(f"saved_csv={sample_metrics_csv_out}")

    with open(anchor_gate_csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dim", "gate_a_anchor", "gate_b_anchor", "gate_a_minus_b_anchor"])
        for j in range(d):
            w.writerow([int(j), float(g_a[anchor_idx, j]), float(g_b[anchor_idx, j]), float(g_a[anchor_idx, j] - g_b[anchor_idx, j])])
    print(f"saved_csv={anchor_gate_csv_out}")

    def _save_group_stats(mask: np.ndarray, out_path: str, group_name: str) -> None:
        idx = np.where(mask)[0]
        if idx.size == 0:
            print(f"csv_warning=empty_group_{group_name}")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["dim", "ga_mean", "ga_std", "gb_mean", "gb_std", "ga_minus_b_mean", "ga_minus_b_std", "count"])
            print(f"saved_csv={out_path}")
            return
        ga = g_a[idx]
        gb = g_b[idx]
        diff = ga - gb
        ga_mean = ga.mean(axis=0)
        ga_std = ga.std(axis=0)
        gb_mean = gb.mean(axis=0)
        gb_std = gb.std(axis=0)
        diff_mean = diff.mean(axis=0)
        diff_std = diff.std(axis=0)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dim", "ga_mean", "ga_std", "gb_mean", "gb_std", "ga_minus_b_mean", "ga_minus_b_std", "count"])
            for j in range(d):
                w.writerow(
                    [
                        int(j),
                        float(ga_mean[j]),
                        float(ga_std[j]),
                        float(gb_mean[j]),
                        float(gb_std[j]),
                        float(diff_mean[j]),
                        float(diff_std[j]),
                        int(idx.size),
                    ]
                )
        print(f"saved_csv={out_path}")

    _save_group_stats(sim_a, sim_a_gate_stats_csv_out, "sim_a")
    _save_group_stats(sim_b, sim_b_gate_stats_csv_out, "sim_b")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if (not args.cpu_only) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device.type}")

    tfm = transforms.ToTensor()
    train_full = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=tfm)

    n_val = 5000
    n_train = len(train_full) - n_val
    train_set, val_set = random_split(
        train_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    print("[two-gate]")
    model = TwoGateDNN(hidden_dims=(args.h1, args.h2, args.h3), gate_hidden=args.gate_hidden, n_classes=2)
    metrics = train_and_eval(model, train_loader, val_loader, test_loader, device, args.epochs, args.lr)
    print(f"two_gate_best_val_acc={metrics['best_val_acc']:.6f}")
    print(f"two_gate_test_acc={metrics['test_acc']:.6f}")

    g_a, g_b, z_space, pred_label, true_digit = collect_spaces_from_loader(
        metrics["model"],
        test_loader,
        device,
        max_samples=args.tsne_samples,
    )

    if 0 <= args.anchor_idx < g_a.shape[0]:
        anchor_idx = int(args.anchor_idx)
    else:
        rng = np.random.default_rng(args.seed)
        anchor_idx = int(rng.integers(0, g_a.shape[0]))
    # g-space_b: gate_a space for all samples, but anchor vector is replaced with gate_b(anchor).
    g_space_b = g_a.copy()
    g_space_b[anchor_idx] = g_b[anchor_idx]

    save_space_tsne_by_pred(g_a, pred_label, args.ga_tsne_out, "g-space_a t-SNE (color=pred label 0/1)")
    save_space_tsne_by_pred(g_space_b, pred_label, args.gb_tsne_out, "g-space_b t-SNE (ga-space, anchor uses gb; color=pred label 0/1)")
    save_space_tsne_by_pred(z_space, pred_label, args.z_tsne_out, "z-space t-SNE (color=pred label 0/1)")
    save_space_tsne_by_digit10(g_a, true_digit, args.ga_tsne_out_digit10, "g-space_a t-SNE (color=true digit 0-9)")
    save_space_tsne_by_digit10(g_space_b, true_digit, args.gb_tsne_out_digit10, "g-space_b t-SNE (ga-space, anchor uses gb; color=true digit 0-9)")
    save_space_tsne_by_digit10(z_space, true_digit, args.z_tsne_out_digit10, "z-space t-SNE (color=true digit 0-9)")

    sim_a, thr_a, adj_a = build_sim_mask(g_a, anchor_idx, args.sim_threshold)
    sim_b, thr_b, adj_b = build_sim_mask_from_anchor_vector(
        g_a,
        g_b[anchor_idx],
        args.sim_threshold,
    )
    anchor_true_digit = int(true_digit[anchor_idx])
    anchor_pred_bin = int(pred_label[anchor_idx])

    print(f"sim_anchor_idx={anchor_idx}")
    print(f"anchor_true_digit_label={anchor_true_digit}")
    print(f"anchor_pred_bin_label={anchor_pred_bin}")
    print("g_space_b_definition=all_ga_except_anchor_is_gb")
    print(f"sim_a_space=g_a")
    print(f"sim_a_cos_threshold_requested={args.sim_threshold:.6f}")
    print(f"sim_a_cos_threshold_effective={thr_a:.6f}")
    print(f"sim_a_cos_threshold_adjusted={int(adj_a)}")
    print(f"sim_a_count={int(sim_a.sum())}")
    print(f"sim_a_ratio={float(sim_a.mean()):.6f}")

    print(f"sim_b_space=g_a")
    print(f"sim_b_anchor_vector=g_b[anchor_idx]")
    print(f"sim_b_cos_threshold_requested={args.sim_threshold:.6f}")
    print(f"sim_b_cos_threshold_effective={thr_b:.6f}")
    print(f"sim_b_cos_threshold_adjusted={int(adj_b)}")
    print(f"sim_b_count={int(sim_b.sum())}")
    print(f"sim_b_ratio={float(sim_b.mean()):.6f}")

    save_space_tsne_with_sim(
        g_a,
        sim_a,
        anchor_idx,
        pred_label,
        args.ga_sim_a_tsne_out,
        f"g-space_a t-SNE (SIM_a from g_a cosine, cos>={thr_a:.4f})",
    )
    save_space_tsne_with_sim(
        g_space_b,
        sim_a,
        anchor_idx,
        pred_label,
        args.gb_sim_a_tsne_out,
        f"g-space_b t-SNE (ga-space, anchor uses gb; SIM_a from g_a cosine, cos>={thr_a:.4f})",
    )
    save_space_tsne_with_sim(
        z_space,
        sim_a,
        anchor_idx,
        pred_label,
        args.z_sim_a_tsne_out,
        f"z-space t-SNE (SIM_a from g_a cosine, cos>={thr_a:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        g_a,
        sim_a,
        anchor_idx,
        true_digit,
        args.ga_sim_a_tsne_out_digit10,
        f"g-space_a t-SNE (SIM_a from g_a cosine, true digit 0-9, cos>={thr_a:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        g_space_b,
        sim_a,
        anchor_idx,
        true_digit,
        args.gb_sim_a_tsne_out_digit10,
        f"g-space_b t-SNE (ga-space, anchor uses gb; SIM_a, true digit 0-9, cos>={thr_a:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        z_space,
        sim_a,
        anchor_idx,
        true_digit,
        args.z_sim_a_tsne_out_digit10,
        f"z-space t-SNE (SIM_a from g_a cosine, true digit 0-9, cos>={thr_a:.4f})",
    )

    save_space_tsne_with_sim(
        g_a,
        sim_b,
        anchor_idx,
        pred_label,
        args.ga_sim_b_tsne_out,
        f"g-space_a t-SNE (SIM_b: cosine(g_a, g_b_anchor), cos>={thr_b:.4f})",
    )
    save_space_tsne_with_sim(
        g_space_b,
        sim_b,
        anchor_idx,
        pred_label,
        args.gb_sim_b_tsne_out,
        f"g-space_b t-SNE (ga-space, anchor uses gb; SIM_b: cosine(g_a, g_b_anchor), cos>={thr_b:.4f})",
    )
    save_space_tsne_with_sim(
        z_space,
        sim_b,
        anchor_idx,
        pred_label,
        args.z_sim_b_tsne_out,
        f"z-space t-SNE (SIM_b: cosine(g_a, g_b_anchor), cos>={thr_b:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        g_a,
        sim_b,
        anchor_idx,
        true_digit,
        args.ga_sim_b_tsne_out_digit10,
        f"g-space_a t-SNE (SIM_b: cosine(g_a, g_b_anchor), true digit 0-9, cos>={thr_b:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        g_space_b,
        sim_b,
        anchor_idx,
        true_digit,
        args.gb_sim_b_tsne_out_digit10,
        f"g-space_b t-SNE (ga-space, anchor uses gb; SIM_b, true digit 0-9, cos>={thr_b:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        z_space,
        sim_b,
        anchor_idx,
        true_digit,
        args.z_sim_b_tsne_out_digit10,
        f"z-space t-SNE (SIM_b: cosine(g_a, g_b_anchor), true digit 0-9, cos>={thr_b:.4f})",
    )

    save_gate_csvs(
        g_a=g_a,
        g_b=g_b,
        pred_label=pred_label,
        true_digit=true_digit,
        anchor_idx=anchor_idx,
        sim_a=sim_a,
        sim_b=sim_b,
        sample_metrics_csv_out=args.sample_gate_metrics_csv_out,
        anchor_gate_csv_out=args.anchor_gate_csv_out,
        sim_a_gate_stats_csv_out=args.sim_a_gate_stats_csv_out,
        sim_b_gate_stats_csv_out=args.sim_b_gate_stats_csv_out,
    )


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    data_candidates = [Path("/workspace/ex/mnist_data"), Path("/workspace/mnist_data")]
    default_data_dir = data_candidates[0] if data_candidates[0].exists() else data_candidates[1]
    default_tag = "exp4_twogate"

    def default_out(name: str) -> str:
        return str(script_dir / f"{name}_{default_tag}.png")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--h1", type=int, default=512)
    parser.add_argument("--h2", type=int, default=256)
    parser.add_argument("--h3", type=int, default=128)
    parser.add_argument("--gate-hidden", type=int, default=256)
    parser.add_argument("--tsne-samples", type=int, default=3000)
    parser.add_argument("--anchor-idx", type=int, default=892)
    parser.add_argument("--sim-threshold", type=float, default=0.85)

    parser.add_argument("--ga-tsne-out", type=str, default=default_out("mnist_ga_space_tsne_pred01"))
    parser.add_argument("--gb-tsne-out", type=str, default=default_out("mnist_gb_space_tsne_pred01"))
    parser.add_argument("--z-tsne-out", type=str, default=default_out("mnist_z_space_tsne_pred01"))
    parser.add_argument("--ga-tsne-out-digit10", type=str, default=default_out("mnist_ga_space_tsne_digit10"))
    parser.add_argument("--gb-tsne-out-digit10", type=str, default=default_out("mnist_gb_space_tsne_digit10"))
    parser.add_argument("--z-tsne-out-digit10", type=str, default=default_out("mnist_z_space_tsne_digit10"))

    parser.add_argument("--ga-sim-a-tsne-out", type=str, default=default_out("mnist_ga_space_tsne_sim_a"))
    parser.add_argument("--gb-sim-a-tsne-out", type=str, default=default_out("mnist_gb_space_tsne_sim_a"))
    parser.add_argument("--z-sim-a-tsne-out", type=str, default=default_out("mnist_z_space_tsne_sim_a"))
    parser.add_argument("--ga-sim-a-tsne-out-digit10", type=str, default=default_out("mnist_ga_space_tsne_sim_a_digit10"))
    parser.add_argument("--gb-sim-a-tsne-out-digit10", type=str, default=default_out("mnist_gb_space_tsne_sim_a_digit10"))
    parser.add_argument("--z-sim-a-tsne-out-digit10", type=str, default=default_out("mnist_z_space_tsne_sim_a_digit10"))

    parser.add_argument("--ga-sim-b-tsne-out", type=str, default=default_out("mnist_ga_space_tsne_sim_b"))
    parser.add_argument("--gb-sim-b-tsne-out", type=str, default=default_out("mnist_gb_space_tsne_sim_b"))
    parser.add_argument("--z-sim-b-tsne-out", type=str, default=default_out("mnist_z_space_tsne_sim_b"))
    parser.add_argument("--ga-sim-b-tsne-out-digit10", type=str, default=default_out("mnist_ga_space_tsne_sim_b_digit10"))
    parser.add_argument("--gb-sim-b-tsne-out-digit10", type=str, default=default_out("mnist_gb_space_tsne_sim_b_digit10"))
    parser.add_argument("--z-sim-b-tsne-out-digit10", type=str, default=default_out("mnist_z_space_tsne_sim_b_digit10"))
    parser.add_argument("--sample-gate-metrics-csv-out", type=str, default=str(script_dir / f"mnist_gate_sample_metrics_{default_tag}.csv"))
    parser.add_argument("--anchor-gate-csv-out", type=str, default=str(script_dir / f"mnist_anchor_gate_values_{default_tag}.csv"))
    parser.add_argument("--sim-a-gate-stats-csv-out", type=str, default=str(script_dir / f"mnist_gate_stats_sim_a_{default_tag}.csv"))
    parser.add_argument("--sim-b-gate-stats-csv-out", type=str, default=str(script_dir / f"mnist_gate_stats_sim_b_{default_tag}.csv"))

    args = parser.parse_args()

    for path in (
        args.ga_tsne_out,
        args.gb_tsne_out,
        args.z_tsne_out,
        args.ga_tsne_out_digit10,
        args.gb_tsne_out_digit10,
        args.z_tsne_out_digit10,
        args.ga_sim_a_tsne_out,
        args.gb_sim_a_tsne_out,
        args.z_sim_a_tsne_out,
        args.ga_sim_a_tsne_out_digit10,
        args.gb_sim_a_tsne_out_digit10,
        args.z_sim_a_tsne_out_digit10,
        args.ga_sim_b_tsne_out,
        args.gb_sim_b_tsne_out,
        args.z_sim_b_tsne_out,
        args.ga_sim_b_tsne_out_digit10,
        args.gb_sim_b_tsne_out_digit10,
        args.z_sim_b_tsne_out_digit10,
        args.sample_gate_metrics_csv_out,
        args.anchor_gate_csv_out,
        args.sim_a_gate_stats_csv_out,
        args.sim_b_gate_stats_csv_out,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    main(args)
