"""The CIFAR-10 split: how it is chosen, recorded, and rebuilt.

The brief forbids submitting dataset downloads, so the images are not in the
repository. What is committed is ``data/split_indices.json``: the 10,000
training and 2,000 validation indices, the seed, per-class counts, a checksum
over the indices, and the normalization statistics. The split is rebuilt from
that, and ``validate_results.py`` checks both runs recorded the same checksum.

Only the official *training* split is ever loaded here. The official test
split is not read anywhere in this project, which is how "kept out of model
selection" is made a property of the code rather than a promise.

Training and validation are drawn from the same 50,000-image pool without
overlap, stratified so every class contributes equally to each.
"""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._config import CLASSES, DATA_DIR, SPLIT_PATH, RunConfig


def select_split(labels, n_train: int, n_val: int, seed: int) -> tuple:
    """Stratified, disjoint (train, val) index arrays, each sorted.

    NumPy's PCG64 rather than torch's generator: the split must not depend on
    which device built it. Validation is drawn first from each class's
    permutation and training follows it, so enlarging ``n_train`` never
    changes the validation set.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if n_train % len(classes) or n_val % len(classes):
        raise ValueError("n_train and n_val must be divisible by the class count")
    per_train, per_val = n_train // len(classes), n_val // len(classes)

    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in classes:
        pool = rng.permutation(np.flatnonzero(labels == c))
        if len(pool) < per_train + per_val:
            raise ValueError(f"class {c} has only {len(pool)} images")
        val.append(pool[:per_val])
        train.append(pool[per_val:per_val + per_train])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def checksum(train_idx, val_idx) -> str:
    """SHA-256 over both index lists. The pixels are fixed by CIFAR-10; what a
    run could get wrong is which of them it trained and validated on."""
    canonical = (
        ",".join(str(int(i)) for i in train_idx) + "|"
        + ",".join(str(int(i)) for i in val_idx)
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def download(root=DATA_DIR):
    """Fetch the official CIFAR-10 *training* split. Never the test split."""
    from torchvision import datasets

    root.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR10(root=str(root), train=True, download=True)


def channel_stats(images_uint8: np.ndarray) -> tuple:
    """Per-channel mean and std of the training subset only, on a [0, 1] scale."""
    x = images_uint8.astype(np.float64) / 255.0
    return x.mean(axis=(0, 1, 2)).tolist(), x.std(axis=(0, 1, 2)).tolist()


def write_split(config: RunConfig, path=SPLIT_PATH) -> dict:
    """Select the split and record it so it can be rebuilt without the data."""
    ds = download()
    train_idx, val_idx = select_split(ds.targets, config.n_train, config.n_val,
                                      config.seed)
    mean, std = channel_stats(ds.data[train_idx])
    targets = np.asarray(ds.targets)
    manifest = {
        "dataset": "CIFAR-10, official training split (50,000 images)",
        "population": len(ds),
        "seed": int(config.seed),
        "n_train": int(config.n_train),
        "n_val": int(config.n_val),
        "selection": ("numpy.random.default_rng(seed); per class: permute, "
                      "first n_val/10 -> val, next n_train/10 -> train; sorted"),
        "test_split_used": False,
        "image_shape": [3, 32, 32],
        "classes": list(CLASSES),
        "train_per_class": np.bincount(targets[train_idx], minlength=10).tolist(),
        "val_per_class": np.bincount(targets[val_idx], minlength=10).tolist(),
        "normalization": ("ToTensor to [0, 1], then (x - mean) / std per channel, "
                          "mean and std computed on the training subset only"),
        "mean": mean,
        "std": std,
        "checksum": checksum(train_idx, val_idx),
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest) + "\n")
    return manifest


def read_split(path=SPLIT_PATH) -> dict:
    if not path.is_file():
        raise SystemExit(
            "No split manifest. Build it first:\n"
            "  .venv/bin/python scripts/build_cifar_split.py"
        )
    return json.loads(path.read_text())


def to_tensor(images_uint8: np.ndarray, mean, std) -> torch.Tensor:
    """(N, 32, 32, 3) uint8 -> (N, 3, 32, 32) float32, normalized."""
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float().div(255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - m) / s


def load_split(config: RunConfig, ds=None) -> dict:
    """Materialize train/val tensors from the committed manifest.

    Returns a dict with ``x_train, y_train, x_val, y_val, raw_val`` (the
    un-normalized validation images, for display), ``val_indices`` and the
    manifest. Everything is held in memory once: 12,000 32x32x3 floats is
    about 147 MB, and removing per-batch decode keeps it out of the timings
    the report has to explain.
    """
    manifest = read_split()
    if (manifest["n_train"], manifest["n_val"], manifest["seed"]) != (
            config.n_train, config.n_val, config.seed):
        raise SystemExit(
            f"Manifest was built for n_train={manifest['n_train']}, "
            f"n_val={manifest['n_val']}, seed={manifest['seed']}; this run asks "
            f"for {config.n_train}, {config.n_val}, {config.seed}. Rebuild it."
        )
    train_idx = np.array(manifest["train_indices"], dtype=np.int64)
    val_idx = np.array(manifest["val_indices"], dtype=np.int64)
    if checksum(train_idx, val_idx) != manifest["checksum"]:
        raise SystemExit("Manifest checksum does not match its own indices.")
    if set(train_idx) & set(val_idx):
        raise SystemExit("Training and validation indices overlap.")

    ds = ds if ds is not None else download()
    targets = np.asarray(ds.targets)
    mean, std = manifest["mean"], manifest["std"]
    return {
        "x_train": to_tensor(ds.data[train_idx], mean, std),
        "y_train": torch.from_numpy(targets[train_idx]).long(),
        "x_val": to_tensor(ds.data[val_idx], mean, std),
        "y_val": torch.from_numpy(targets[val_idx]).long(),
        "raw_val": ds.data[val_idx],
        "raw_train": ds.data[train_idx],
        "train_indices": train_idx,
        "val_indices": val_idx,
        "manifest": manifest,
    }


def make_loader(x, y, batch_size: int, seed: int) -> DataLoader:
    """A shuffling loader whose order is the same on every device.

    The generator lives on the CPU: torch's MPS and CPU streams differ, and
    the batch order should not change because a run moved to another machine.
    Both patch sizes therefore see identical batches in identical order.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    from torch.utils.data import TensorDataset

    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                      drop_last=False, generator=generator)


def fixed_example_positions(seed: int, n_val: int, n: int, labels) -> list:
    """Positions in the validation array of the fixed display examples.

    Chosen from the seed and the labels alone - never from any model's
    predictions - so both patch sizes are shown on the same images and the
    choice cannot be tuned to flatter either. One example per class for ``n``
    distinct classes, drawn at random.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed + 1)
    picked = rng.choice(10, size=n, replace=False)
    return [int(rng.choice(np.flatnonzero(labels == c))) for c in sorted(picked)]
