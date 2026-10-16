"""Figures. Each takes saved run data (and images where needed) and a path."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._config import CLASSES  # noqa: E402

COLORS = {"p4": "#1f77b4", "p8": "#d95f02"}


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def sample_grid(images, labels, path, n=16, title="Sixteen labeled training examples"):
    fig, axes = plt.subplots(2, n // 2, figsize=(n / 2 * 1.2, 3.2))
    for ax, img, lab in zip(axes.ravel(), images[:n], labels[:n]):
        ax.imshow(img)
        ax.set_title(CLASSES[int(lab)], fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    _save(fig, path)


def learning_curves(run, name, path):
    h = run["history"]
    epochs = [r["epoch"] for r in h]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, label in ((a, "loss", "Loss"), (b, "acc", "Accuracy")):
        ax.plot(epochs, [r[f"train_{key}"] for r in h], "o-", label="train (dropout on, running)")
        ax.plot(epochs, [r[f"val_{key}"] for r in h], "s-", label="validation")
        ax.set_xlabel("epoch"); ax.set_ylabel(label); ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.suptitle(f"{name}: {run['config']['patch_size']}x{run['config']['patch_size']} patches")
    _save(fig, path)


def comparison_curves(runs, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, run in runs.items():
        h = run["history"]
        e = [r["epoch"] for r in h]
        p = run["config"]["patch_size"]
        for ax, key in ((a, "loss"), (b, "acc")):
            ax.plot(e, [r[f"train_{key}"] for r in h], "--", color=COLORS[name], alpha=0.6,
                    label=f"{p}x{p} train")
            ax.plot(e, [r[f"val_{key}"] for r in h], "-o", color=COLORS[name],
                    label=f"{p}x{p} validation")
    for ax, label in ((a, "Loss"), (b, "Accuracy")):
        ax.set_xlabel("epoch"); ax.set_ylabel(label); ax.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.suptitle("Same split, seed and settings; only the patch size differs")
    _save(fig, path)


def predictions_panel(run, raw_val, path, title=None):
    ex = run["fixed_examples"]
    fig, axes = plt.subplots(1, len(ex), figsize=(len(ex) * 1.5, 2.3))
    for ax, e in zip(axes, ex):
        ax.imshow(raw_val[e["position"]])
        ok = e["true"] == e["pred"]
        ax.set_title(f"true: {e['true']}\npred: {e['pred']}\n{e['prob']:.2f}", fontsize=7,
                     color="#1a7f37" if ok else "#c62828")
        ax.axis("off")
    fig.suptitle(title or f"{run['config']['name']}: fixed validation examples "
                 "(green = correct, red = error; number = confidence)", fontsize=9)
    _save(fig, path)


def attention_cost(runs, path):
    names = list(runs)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.2))
    labels = [f"{runs[n]['config']['patch_size']}x{runs[n]['config']['patch_size']}" for n in names]
    a.bar(labels, [runs[n]["model"]["attn_entries"] for n in names],
          color=[COLORS[n] for n in names])
    a.set_title("L² score entries per head per layer", fontsize=9)
    b.bar(labels, [runs[n]["total_seconds"] for n in names], color=[COLORS[n] for n in names])
    b.set_title("Measured training + validation time (s)", fontsize=9)
    for ax in (a, b):
        ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def per_class_accuracy(runs, path):
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(CLASSES))
    for i, (name, run) in enumerate(runs.items()):
        right = np.zeros(10)
        count = np.zeros(10)
        for t, p in zip(run["val_labels"], run["val_preds"]):
            count[t] += 1
            right[t] += t == p
        p_ = run["config"]["patch_size"]
        ax.bar(x + (i - 0.5) * 0.38, right / count, 0.38, color=COLORS[name],
               label=f"{p_}x{p_}")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("validation accuracy")
    ax.set_title("Per-class validation accuracy, seed 42 (200 images per class)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def seed_replication(by_run, path):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, row in by_run.items():
        p = {"p4": 4, "p8": 8}[name]
        ax.plot(row["seeds"], row["val_acc"], "o-", color=COLORS[name], label=f"{p}x{p}")
    ax.set_xlabel("seed (same split, same settings)")
    ax.set_ylabel("final validation accuracy")
    ax.set_xticks(by_run["p4"]["seeds"])
    ax.set_title("Seed replication", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)
