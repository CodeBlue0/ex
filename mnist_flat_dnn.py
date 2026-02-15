from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
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


class FlatDNN(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden_dims: tuple[int, ...] = (512, 256, 128), n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dims[2], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten image -> tabular-like input
        z = self.features(x)
        return self.classifier(z)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.features(x)


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


@torch.no_grad()
def save_tsne_10color(model: nn.Module, loader: DataLoader, device: torch.device, out_path: str, max_samples: int = 3000) -> None:
    model.eval()
    feats = []
    labels = []
    seen = 0
    for xb, y_digit in loader:
        xb = xb.to(device)
        z = model.embed(xb).detach().cpu().numpy()
        feats.append(z)
        labels.append(y_digit.numpy())
        seen += len(xb)
        if seen >= max_samples:
            break

    X = np.concatenate(feats, axis=0)[:max_samples]
    y = np.concatenate(labels, axis=0)[:max_samples]

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        m = y == d
        if np.any(m):
            plt.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.75, color=cmap(d), label=str(d))
    plt.title("MNIST t-SNE (10-color by original digit)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="digit", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"saved_tsne={out_path}")


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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = FlatDNN(hidden_dims=(args.h1, args.h2, args.h3), n_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
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
        print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate_binary(model, test_loader, device)
    print("task=binary_0to4_vs_5to9")
    print(f"best_val_acc={best_val_acc:.6f}")
    print(f"test_loss={test_loss:.6f}")
    print(f"test_acc={test_acc:.6f}")

    save_tsne_10color(
        model=model,
        loader=test_loader,
        device=device,
        out_path=args.tsne_out,
        max_samples=args.tsne_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/workspace/mnist_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--h1", type=int, default=512)
    parser.add_argument("--h2", type=int, default=256)
    parser.add_argument("--h3", type=int, default=128)
    parser.add_argument("--tsne-out", type=str, default="/workspace/mnist_tsne_10color.png")
    parser.add_argument("--tsne-samples", type=int, default=3000)
    args = parser.parse_args()
    main(args)
