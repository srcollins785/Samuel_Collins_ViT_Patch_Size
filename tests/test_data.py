"""The split: determinism, stratification, disjointness, checksum sensitivity.

Uses synthetic labels, so no CIFAR download is needed.
"""

import numpy as np
import pytest

from vit_patch_size import data

LABELS = np.repeat(np.arange(10), 5000)   # 50,000 labels, 5,000 per class


def split(seed=42, n_train=10_000, n_val=2_000):
    return data.select_split(LABELS, n_train, n_val, seed)


def test_sizes_and_stratification():
    train, val = split()
    assert len(train) == 10_000 and len(val) == 2_000
    assert np.bincount(LABELS[train]).tolist() == [1000] * 10
    assert np.bincount(LABELS[val]).tolist() == [200] * 10


def test_disjoint_sorted_unique():
    train, val = split()
    assert not set(train) & set(val)
    assert len(set(train)) == len(train)
    assert (np.diff(train) > 0).all() and (np.diff(val) > 0).all()


def test_deterministic_and_seed_sensitive():
    a, b = split(), split()
    assert all((x == y).all() for x, y in zip(a, b))
    c = split(seed=7)
    assert not (a[0] == c[0]).all()


def test_validation_set_does_not_depend_on_training_size():
    _, val_big = split(n_train=10_000)
    _, val_small = split(n_train=2_000)
    assert (val_big == val_small).all()


def test_checksum_changes_if_any_index_changes():
    train, val = split()
    base = data.checksum(train, val)
    tweaked = train.copy()
    tweaked[0] += 1
    assert data.checksum(tweaked, val) != base
    assert data.checksum(val, train) != base   # order of the two lists matters


def test_indivisible_sizes_rejected():
    with pytest.raises(ValueError):
        data.select_split(LABELS, 10_005, 2_000, 42)


def test_fixed_examples_depend_on_labels_and_seed_only():
    labels = LABELS[split()[1]]
    a = data.fixed_example_positions(42, 2000, 8, labels)
    assert a == data.fixed_example_positions(42, 2000, 8, labels)
    assert len(a) == 8 and len(set(labels[a])) == 8     # eight distinct classes


def test_normalization_statistics_are_per_channel():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(50, 32, 32, 3), dtype=np.uint8)
    mean, std = data.channel_stats(imgs)
    x = data.to_tensor(imgs, mean, std)
    assert x.shape == (50, 3, 32, 32)
    assert x.mean(dim=(0, 2, 3)).abs().max() < 1e-4
    assert (x.std(dim=(0, 2, 3), unbiased=False) - 1).abs().max() < 1e-3
