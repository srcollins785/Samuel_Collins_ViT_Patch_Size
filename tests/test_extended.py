"""The CNN reference and the early-stopping logic."""

import torch

from vit_patch_size import RunConfig, extended
from vit_patch_size.cnn import SmallCNN
from vit_patch_size.model import count_parameters

VIT = {"p4": 75_082, "p8": 81_226}


def test_cnn_is_parameter_matched_to_the_4x4_vit():
    n = count_parameters(SmallCNN())
    assert n == 75_096
    assert abs(n - VIT["p4"]) / VIT["p4"] < 0.001      # within 0.1%


def test_cnn_shapes_and_logits():
    m = SmallCNN().eval()
    x = torch.randn(3, 3, 32, 32)
    assert m.features(x).shape == (3, 86, 8, 8)
    assert m(x).shape == (3, 10)


def test_cnn_dropout_is_off_in_eval_and_batchnorm_uses_running_stats():
    m = SmallCNN()
    x = torch.randn(8, 3, 32, 32)
    m.train()
    assert not torch.equal(m(x), m(x))
    m.eval()
    assert torch.equal(m(x), m(x))


def test_should_stop_needs_patience_epochs_without_a_new_best():
    assert not extended.should_stop([1.0, 0.9, 0.8], patience=3)
    assert not extended.should_stop([1.0, 0.8, 0.9, 0.9], patience=3)      # 2 since best
    assert extended.should_stop([1.0, 0.8, 0.9, 0.9, 0.9], patience=3)     # 3 since best
    assert not extended.should_stop([1.0, 0.8, 0.9, 0.7], patience=3)      # new best resets
    assert not extended.should_stop([], patience=3)


def test_arms_are_built_from_the_same_seed():
    cfg = RunConfig()
    a, b = extended.build_arm("p4", cfg), extended.build_arm("p4", cfg)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert count_parameters(extended.build_arm("p8", cfg)) == VIT["p8"]


def test_run_records_every_epoch_and_the_best_one():
    g = torch.Generator().manual_seed(0)
    split = {"x_train": torch.randn(64, 3, 32, 32, generator=g),
             "y_train": torch.randint(0, 10, (64,), generator=g),
             "x_val": torch.randn(32, 3, 32, 32, generator=g),
             "y_val": torch.randint(0, 10, (32,), generator=g)}
    cfg = RunConfig(device="cpu", n_train=64, n_val=32)
    r = extended.run("cnn", cfg, split, max_epochs=4, patience=2, log=lambda *_: None)
    assert 1 <= r["epochs_run"] <= 4 and len(r["history"]) == r["epochs_run"]
    losses = [h["val_loss"] for h in r["history"]]
    assert r["best_epoch"] == losses.index(min(losses)) + 1
    assert r["val_acc_at_best"] == r["history"][r["best_epoch"] - 1]["val_acc"]
