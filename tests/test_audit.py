"""The audit must accept a coherent pair of runs and reject each way of breaking one."""

import copy
import json

import pytest

import validate_results as vr
from test_report_data import make_ext, make_run


def coherent(patch):
    n, length, entries = (32 // patch) ** 2, (32 // patch) ** 2 + 1, ((32 // patch) ** 2 + 1) ** 2
    params = {4: 75_082, 8: 81_226}[patch]
    run = make_run(patch, [0.3, 0.4, 0.5], 5.0)
    run["config"].update(optimizer="Adam", loss="ce", normalization="n",
                         normalization_mean=[0, 0, 0], normalization_std=[1, 1, 1],
                         betas=[0.9, 0.999], weight_decay=0.0, n_fixed_examples=8,
                         split_checksum="abc", name=f"p{patch}")
    run["model"].update(
        patch_size=patch, num_patches=n, tokens=length, attn_entries=entries,
        attn_entries_total=entries * 8, parameters=params,
        parameter_breakdown={"all": params},
        shapes={"patches": [2, n, 3 * patch * patch], "logits": [2, 10],
                "with_position_embedding": [2, length, 64]})
    run["final_val_acc"] = 0.5
    run["val_labels"] = [0, 1, 2, 3]
    run["val_preds"] = [0, 1, 0, 0]       # 2 of 4 right... fixed below
    run["val_preds"] = [0, 1, 2, 0]
    run["final_val_acc"] = 0.75
    run["history"][-1]["val_acc"] = 0.75
    return run


@pytest.fixture
def setup(tmp_path):
    def write(mutate=None, extended_mutate=None):
        runs = {"p4": coherent(4), "p8": coherent(8)}
        if mutate:
            mutate(runs)
        for k, r in runs.items():
            (tmp_path / f"{k}.json").write_text(json.dumps(r))
        make_ext(tmp_path, graded=runs, epochs=40, patience=2)
        if extended_mutate:
            path = tmp_path / "supplementary" / "extended.json"
            d = json.loads(path.read_text())
            extended_mutate(d)
            path.write_text(json.dumps(d))
        split = tmp_path / "split.json"
        split.write_text(json.dumps({
            "checksum": "abc", "test_split_used": False,
            "train_indices": list(range(10_000)),
            "val_indices": list(range(10_000, 12_000))}))
        return vr.audit(tmp_path, split)
    return write


def test_coherent_runs_pass(setup):
    assert setup() == []


@pytest.mark.parametrize("name,mutate,expect", [
    ("lr differs", lambda r: r["p8"]["config"].update(lr=1e-3), "lr"),
    ("dropout differs", lambda r: r["p8"]["config"].update(dropout=0.0), "dropout"),
    ("different split", lambda r: r["p8"]["config"].update(split_checksum="zzz"), "split_checksum"),
    ("wrong param count", lambda r: r["p4"]["model"].update(parameters=1), "parameters"),
    ("bad token count", lambda r: r["p8"]["model"].update(tokens=65), "token"),
    ("different examples", lambda r: r["p8"]["fixed_examples"][0].update(index=99), "different fixed"),
    ("stale accuracy", lambda r: r["p4"].update(final_val_acc=0.9), "final_val_acc"),
    ("runtime not summed", lambda r: r["p4"].update(total_seconds=1.0), "total_seconds"),
])
def test_each_corruption_is_caught(setup, name, mutate, expect):
    problems = setup(mutate)
    assert any(expect in p for p in problems), (name, problems)


def run_of(d, arm, seed=42):
    return next(r for r in d["runs"] if r["arm"] == arm and r["seed"] == seed)


@pytest.mark.parametrize("name,mutate,expect", [
    ("cnn not parameter matched", lambda d: run_of(d, "cnn").update(parameters=60_000), "CNN has"),
    ("wrong best epoch", lambda d: run_of(d, "p4").update(best_epoch=3), "minimum-validation-loss"),
    ("val_acc_at_best stale", lambda d: run_of(d, "p8").update(val_acc_at_best=0.99), "val_acc_at_best"),
    ("different split", lambda d: d.update(split_checksum="zzz"), "different split"),
    ("seed-42 curve differs from graded", lambda d: run_of(d, "p4")["history"][0].update(val_acc=0.123),
     "reproduce the graded"),
    ("stopped at the wrong time", lambda d: run_of(d, "cnn").update(
        history=run_of(d, "cnn")["history"][:-1], epochs_run=11), "patience"),
])
def test_each_extended_corruption_is_caught(setup, name, mutate, expect):
    problems = setup(extended_mutate=mutate)
    assert any(expect in p for p in problems), (name, problems)
