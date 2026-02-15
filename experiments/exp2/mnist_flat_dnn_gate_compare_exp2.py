from __future__ import annotations

import argparse
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


class BaselineDNN(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden_dims: tuple[int, ...] = (512, 256, 128), n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


class GatedDNN(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden_dims: tuple[int, ...] = (512, 256, 128), gate_hidden: int = 256, n_classes: int = 2):
        super().__init__()
        self.gate_net = nn.Sequential(
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

    def forward_with_spaces(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        g = torch.softmax(self.gate_net(x), dim=1)
        # Pixelwise gating: each input dimension is scaled by its own gate weight.
        z = x * g
        logits = self.backbone(z)
        return logits, g, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_spaces(x)
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
def save_space_tsne_by_pred(space: np.ndarray, pred_label: np.ndarray, out_path: str, title: str) -> None:
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    emb = tsne.fit_transform(space)
    plt.figure(figsize=(8, 6))
    cmap = np.array(["#1f77b4", "#d62728"])
    plt.scatter(emb[:, 0], emb[:, 1], c=cmap[pred_label], s=8, alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved_tsne={out_path}")


def save_key_space_tsne(
    g_space_for_sim: np.ndarray,
    z_space: np.ndarray,
    pred_label: np.ndarray,
    anchor_idx: int,
    key_topk: int,
    sim_cos_threshold: float,
    min_sim_count: int,
    out_path: str,
    key_dims: list[int] | None = None,
) -> None:
    anchor_g = g_space_for_sim[anchor_idx]
    if key_dims is not None and len(key_dims) > 0:
        key_idx = np.array(key_dims, dtype=np.int64)
        key_idx = key_idx[(key_idx >= 0) & (key_idx < z_space.shape[1])]
        if key_idx.size == 0:
            key_idx = np.array([int(np.argmax(anchor_g))], dtype=np.int64)
            print("key_space_warning=invalid_fixed_dims_fallback_to_top1")
        key_mode = "fixed_dims"
    else:
        k = max(1, min(int(key_topk), z_space.shape[1]))
        key_idx = np.argsort(anchor_g)[-k:][::-1].astype(np.int64)
        key_mode = "anchor_topk"

    key_space = z_space[:, key_idx]
    c = cosine_similarity_to_anchor(g_space_for_sim, anchor_idx)
    sim_mask = c >= sim_cos_threshold
    effective_threshold = sim_cos_threshold
    requested_min = max(1, min(int(min_sim_count), c.shape[0]))
    if int(sim_mask.sum()) < requested_min:
        kth = float(np.partition(c, c.shape[0] - requested_min)[c.shape[0] - requested_min])
        effective_threshold = min(sim_cos_threshold, kth)
        sim_mask = c >= effective_threshold
    save_space_tsne_with_sim(
        space=key_space,
        sim_mask=sim_mask,
        anchor_idx=anchor_idx,
        label=pred_label,
        out_path=out_path,
        title=(
            f"key-space t-SNE (anchor top-{int(key_idx.size)} dims, "
            f"SIM cos>={effective_threshold:.4f})"
        ),
    )

    print(f"key_space_mode={key_mode}")
    print(f"key_space_topk={int(key_idx.size)}")
    print(f"key_space_sim_cos_threshold_effective={effective_threshold:.6f}")
    print(f"key_space_anchor_idx={anchor_idx}")
    print(f"key_space_dims={key_idx.tolist()}")
    print(f"key_space_dim={int(key_idx.size)}")


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

    plt.figure(figsize=(8, 6))
    non = ~sim_mask
    cmap = np.array(["#1f77b4", "#d62728"])
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

    plt.figure(figsize=(8, 6))
    non = ~sim_mask
    cmap = plt.get_cmap("tab10")
    c_non = cmap(np.mod(digit_label[non], 10))
    c_sim = cmap(np.mod(digit_label[sim_mask], 10))
    plt.scatter(emb[non, 0], emb[non, 1], c=c_non, s=7, alpha=0.10)
    plt.scatter(emb[sim_mask, 0], emb[sim_mask, 1], c=c_sim, s=11, alpha=0.90)
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

    digit_handles = [
        Line2D([], [], marker="o", linestyle="", color=cmap(i), markersize=6, label=str(i))
        for i in range(10)
    ]
    plt.legend(handles=digit_handles, loc="lower right", title="Digit", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved_tsne={out_path}")


def save_gz_tsne_with_gspace_sim(
    g_space: np.ndarray,
    g_space_for_sim: np.ndarray,
    z_space: np.ndarray,
    pred_label: np.ndarray,
    digit_label: np.ndarray,
    anchor_idx: int,
    g_out: str,
    z_out: str,
    g_out_digit10: str,
    z_out_digit10: str,
    cos_threshold: float,
    min_sim_count: int,
) -> None:
    c = cosine_similarity_to_anchor(g_space_for_sim, anchor_idx)
    sim_mask = c >= cos_threshold
    effective_threshold = cos_threshold
    adjusted_threshold = False

    requested_min = max(1, min(int(min_sim_count), c.shape[0]))
    if int(sim_mask.sum()) < requested_min:
        kth = float(np.partition(c, c.shape[0] - requested_min)[c.shape[0] - requested_min])
        effective_threshold = min(cos_threshold, kth)
        sim_mask = c >= effective_threshold
        adjusted_threshold = True

    print(f"sim_anchor_idx={anchor_idx}")
    print(f"sim_on_log_centered_g={int(np.any(np.abs(g_space_for_sim - g_space) > 1e-12))}")
    print(f"sim_cos_threshold_requested={cos_threshold:.6f}")
    print(f"sim_cos_threshold_effective={effective_threshold:.6f}")
    print(f"sim_cos_threshold_adjusted={int(adjusted_threshold)}")
    print(f"sim_min_count_target={requested_min}")
    print(f"sim_count={int(sim_mask.sum())}")
    print(f"sim_ratio={float(sim_mask.mean()):.6f}")
    save_space_tsne_with_sim(
        space=g_space_for_sim,
        sim_mask=sim_mask,
        anchor_idx=anchor_idx,
        label=pred_label,
        out_path=g_out,
        title=f"log-centered g-space t-SNE (SIM from cosine on log-centered g, cos>={effective_threshold:.4f})",
    )
    save_space_tsne_with_sim(
        space=z_space,
        sim_mask=sim_mask,
        anchor_idx=anchor_idx,
        label=pred_label,
        out_path=z_out,
        title=f"z-space t-SNE (SIM from cosine on log-centered g, cos>={effective_threshold:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        space=g_space_for_sim,
        sim_mask=sim_mask,
        anchor_idx=anchor_idx,
        digit_label=digit_label,
        out_path=g_out_digit10,
        title=f"log-centered g-space t-SNE (SIM from cosine, true digit 0-9, cos>={effective_threshold:.4f})",
    )
    save_space_tsne_with_sim_digit10(
        space=z_space,
        sim_mask=sim_mask,
        anchor_idx=anchor_idx,
        digit_label=digit_label,
        out_path=z_out_digit10,
        title=f"z-space t-SNE (SIM from cosine, true digit 0-9, cos>={effective_threshold:.4f})",
    )


@torch.no_grad()
def collect_gz_from_loader(
    model: GatedDNN,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 3000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    g_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    digit_list: list[np.ndarray] = []
    collected = 0

    for xb, y_digit in loader:
        xb = xb.to(device)
        logits, g, z = model.forward_with_spaces(xb)
        g_np = g.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        p_np = logits.argmax(dim=1).detach().cpu().numpy()
        d_np = y_digit.detach().cpu().numpy()

        if collected + g_np.shape[0] > max_samples:
            keep = max_samples - collected
            if keep <= 0:
                break
            g_np = g_np[:keep]
            z_np = z_np[:keep]
            p_np = p_np[:keep]
            d_np = d_np[:keep]

        g_list.append(g_np)
        z_list.append(z_np)
        pred_list.append(p_np)
        digit_list.append(d_np)
        collected += g_np.shape[0]
        if collected >= max_samples:
            break

    return (
        np.concatenate(g_list, axis=0),
        np.concatenate(z_list, axis=0),
        np.concatenate(pred_list, axis=0),
        np.concatenate(digit_list, axis=0),
    )


def log_center_g_space(g_space: np.ndarray, eps: float) -> np.ndarray:
    l = np.log(g_space + eps)
    return l - l.mean(axis=1, keepdims=True)


def l2_normalize_rows(space: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(space, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return space / norms


def cosine_similarity_to_anchor(space: np.ndarray, anchor_idx: int) -> np.ndarray:
    anchor = space[anchor_idx]
    anchor_norm = np.linalg.norm(anchor)
    norms = np.linalg.norm(space, axis=1)
    denom = np.maximum(norms * max(anchor_norm, 1e-12), 1e-12)
    return (space @ anchor) / denom


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if (not args.cpu_only and torch.cuda.is_available()) else "cpu")
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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    print("[gated]")
    gate = GatedDNN(hidden_dims=(args.h1, args.h2, args.h3), gate_hidden=args.gate_hidden, n_classes=2)
    m_gate = train_and_eval(gate, train_loader, val_loader, test_loader, device, args.epochs, args.lr)
    print(f"gated_best_val_acc={m_gate['best_val_acc']:.6f}")
    print(f"gated_test_acc={m_gate['test_acc']:.6f}")
    g_space, z_space, pred_label, true_digit = collect_gz_from_loader(
        m_gate["model"],
        test_loader,
        device,
        max_samples=args.tsne_samples,
    )
    if not args.no_log_center_g_for_sim:
        g_space_for_sim = l2_normalize_rows(log_center_g_space(g_space, args.log_eps))
    else:
        g_space_for_sim = g_space
    save_space_tsne_by_pred(
        space=g_space_for_sim,
        pred_label=pred_label,
        out_path=args.g_tsne_out,
        title=(
            f"g-space t-SNE (log-center, eps={args.log_eps:.1e}, color=pred 0/1)"
            if (not args.no_log_center_g_for_sim)
            else "g-space t-SNE (raw, color=pred 0/1)"
        ),
    )
    save_space_tsne_by_pred(
        space=z_space,
        pred_label=pred_label,
        out_path=args.z_tsne_out,
        title="z-space t-SNE (color=pred label 0/1)",
    )
    rng = np.random.default_rng(args.seed)
    anchor_idx = int(rng.integers(0, g_space.shape[0]))

    save_gz_tsne_with_gspace_sim(
        g_space=g_space,
        g_space_for_sim=g_space_for_sim,
        z_space=z_space,
        pred_label=pred_label,
        digit_label=true_digit,
        anchor_idx=anchor_idx,
        g_out=args.g_sim_tsne_out,
        z_out=args.z_sim_tsne_out,
        g_out_digit10=args.g_sim_tsne_out_digit10,
        z_out_digit10=args.z_sim_tsne_out_digit10,
        cos_threshold=args.sim_threshold,
        min_sim_count=args.min_sim_count,
    )
    save_key_space_tsne(
        g_space_for_sim=g_space_for_sim,
        z_space=z_space,
        pred_label=pred_label,
        anchor_idx=anchor_idx,
        key_topk=args.key_topk,
        sim_cos_threshold=args.sim_threshold,
        min_sim_count=args.min_sim_count,
        out_path=args.key_space_tsne_out,
        key_dims=parse_key_dims(args.key_dims),
    )

    print("[baseline] skipped in exp2")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_data_dir = Path("/workspace/mnist_data")
    default_key_dims = ""
    default_tag = "exp2_logc"

    def default_out(name: str) -> str:
        return str(script_dir / f"{name}_{default_tag}.png")

    def parse_key_dims(text: str) -> list[int] | None:
        t = text.strip()
        if not t:
            return None
        out: list[int] = []
        for part in t.split(","):
            p = part.strip()
            if p:
                out.append(int(p))
        return out

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--h1", type=int, default=512)
    parser.add_argument("--h2", type=int, default=256)
    parser.add_argument("--h3", type=int, default=128)
    parser.add_argument("--gate-hidden", type=int, default=256)
    parser.add_argument("--g-tsne-out", type=str, default=default_out("mnist_g_space_tsne_pred01"))
    parser.add_argument("--z-tsne-out", type=str, default=default_out("mnist_z_space_tsne_pred01"))
    parser.add_argument("--g-sim-tsne-out", type=str, default=default_out("mnist_g_space_tsne_sim_from_g"))
    parser.add_argument("--z-sim-tsne-out", type=str, default=default_out("mnist_z_space_tsne_sim_from_g"))
    parser.add_argument("--g-sim-tsne-out-digit10", type=str, default=default_out("mnist_g_space_tsne_sim_from_g_digit10"))
    parser.add_argument("--z-sim-tsne-out-digit10", type=str, default=default_out("mnist_z_space_tsne_sim_from_g_digit10"))
    parser.add_argument("--key-space-tsne-out", type=str, default=default_out("mnist_key_space_tsne_top3"))
    parser.add_argument("--key-dims", type=str, default=default_key_dims)
    parser.add_argument("--key-topk", type=int, default=3)
    parser.add_argument("--tsne-samples", type=int, default=3000)
    parser.add_argument("--sim-threshold", type=float, default=0.998)
    parser.add_argument("--min-sim-count", type=int, default=30)
    parser.add_argument("--log-eps", type=float, default=1e-8)
    parser.add_argument("--no-log-center-g-for-sim", action="store_true")
    args = parser.parse_args()

    for path in (
        args.g_tsne_out,
        args.z_tsne_out,
        args.g_sim_tsne_out,
        args.z_sim_tsne_out,
        args.g_sim_tsne_out_digit10,
        args.z_sim_tsne_out_digit10,
        args.key_space_tsne_out,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    main(args)
