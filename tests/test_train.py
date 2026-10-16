"""Training and evaluation behave as the report says they do."""

import copy

import torch
from torch import nn

from vit_patch_size import RunConfig, COMPARED_FIELDS
from vit_patch_size import train as T
from vit_patch_size.evaluate import evaluate


def tiny_split(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)


def test_evaluation_changes_no_parameter_and_is_deterministic():
    cfg = RunConfig()
    model = T.build_model(cfg)
    model.train()
    before = copy.deepcopy(model.state_dict())
    x, y = tiny_split()
    a = evaluate(model, x, y, torch.device("cpu"))
    b = evaluate(model, x, y, torch.device("cpu"))
    assert a["loss"] == b["loss"] and a["preds"] == b["preds"]   # dropout is off
    assert model.training                                          # mode restored
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_training_mode_is_stochastic_evaluation_mode_is_not():
    model = T.build_model(RunConfig())
    x, _ = tiny_split(8)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))


def test_loss_is_cross_entropy_on_raw_logits():
    """CrossEntropyLoss = log_softmax + NLL. Feeding it softmax output would
    be a second softmax; the two give different values."""
    logits = torch.tensor([[2.0, 0.5, -1.0]])
    y = torch.tensor([0])
    ce = nn.CrossEntropyLoss()
    right = ce(logits, y)
    manual = -torch.log(logits.softmax(-1)[0, 0])
    assert torch.allclose(right, manual)
    assert not torch.allclose(ce(logits.softmax(-1), y), right)


def test_optimizer_step_changes_the_parameters():
    cfg = RunConfig()
    model = T.build_model(cfg)
    x, y = tiny_split(16)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    before = [p.detach().clone() for p in model.parameters()]
    loader = [(x, y)]
    T.train_epoch(model, loader, opt, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_same_seed_same_initial_weights_and_batch_order():
    a, b = T.build_model(RunConfig()), T.build_model(RunConfig())
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    from vit_patch_size import data
    x, y = tiny_split(40)
    o1 = [yb.tolist() for _, yb in data.make_loader(x, y, 8, 42)]
    o2 = [yb.tolist() for _, yb in data.make_loader(x, y, 8, 42)]
    assert o1 == o2


def test_the_two_configs_differ_only_in_patch_size():
    a = RunConfig()
    b = a.with_patch(8)
    for field in COMPARED_FIELDS:
        assert getattr(a, field) == getattr(b, field), field
    differing = {k for k, v in a.to_dict().items() if b.to_dict()[k] != v}
    assert differing == {"name", "patch_size"}


def test_end_to_end_run_produces_the_documented_schema():
    import report_data as rd
    cfg = RunConfig(epochs=2, n_train=64, n_val=32, device="cpu")
    split = {
        "x_train": tiny_split(64)[0], "y_train": tiny_split(64)[1],
        "x_val": tiny_split(32, 1)[0], "y_val": tiny_split(32, 1)[1],
        "val_indices": list(range(32)),
        "manifest": {"checksum": "c", "normalization": "n",
                     "mean": [0, 0, 0], "std": [1, 1, 1]},
    }
    result = T.run(cfg, split, log=lambda *_: None)
    rd._check("p4", result)
    assert result["model"]["parameters"] == 75_082
    assert result["model"]["attn_entries"] == 4225
    assert len(result["fixed_examples"]) == 8
