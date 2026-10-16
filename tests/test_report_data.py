"""The report boundary: a synthetic pair of run files exercises every query."""

import json

import pytest

import report_data as rd


def make_run(patch: int, val_accs, seconds_per_epoch: float) -> dict:
    n, length, entries = rd.tokens(patch)
    return {
        "config": {
            "patch_size": patch, "embed_dim": 64, "depth": 2, "heads": 4,
            "mlp_dim": 128, "dropout": 0.1, "lr": 3e-4, "batch_size": 64,
            "epochs": len(val_accs), "seed": 42, "n_train": 10000,
            "n_val": 2000, "split_checksum": "abc",
        },
        "model": {"num_patches": n, "tokens": length, "attn_entries": entries,
                  "parameters": 1},
        "history": [
            {"epoch": i + 1, "train_loss": 2.0 - 0.1 * i, "train_acc": 0.2,
             "val_loss": 2.0 - 0.1 * i, "val_acc": a, "seconds": seconds_per_epoch}
            for i, a in enumerate(val_accs)
        ],
        "total_seconds": seconds_per_epoch * len(val_accs),
        "fixed_examples": [
            {"index": i, "true": "cat", "pred": "cat" if i % 2 else "dog",
             "prob": 0.5}
            for i in range(8)
        ],
        "hardware": {"device": "mps"},
        "software": {"torch": "x"},
    }


@pytest.fixture
def results(tmp_path):
    for name, patch, accs, secs in (("p4", 4, [0.3, 0.4, 0.45], 10.0),
                                    ("p8", 8, [0.3, 0.38, 0.42], 4.0)):
        (tmp_path / f"{name}.json").write_text(json.dumps(make_run(patch, accs, secs)))
    return tmp_path


def test_token_arithmetic_matches_the_brief():
    assert rd.tokens(4) == (64, 65, 4225)
    assert rd.tokens(8) == (16, 17, 289)


def test_queries(results):
    data = rd.load(results)
    assert rd.higher_final_val_acc(data) == "p4"
    assert rd.val_acc_gap(data) == pytest.approx(0.03)
    assert rd.attention_entry_ratio(data) == pytest.approx(4225 / 289)
    assert rd.runtime_ratio(data) == pytest.approx(2.5)
    assert rd.same_fixed_examples(data)
    assert rd.gap_exceeds_seed_noise(data) is None  # no replication: say so


def test_incomplete_run_is_rejected(results):
    run = json.loads((results / "p4.json").read_text())
    run["history"].pop()
    (results / "p4.json").write_text(json.dumps(run))
    with pytest.raises(SystemExit, match="incomplete"):
        rd.load(results)


def test_missing_key_is_named(results):
    run = json.loads((results / "p8.json").read_text())
    del run["config"]["seed"]
    (results / "p8.json").write_text(json.dumps(run))
    with pytest.raises(SystemExit, match="config.seed"):
        rd.load(results)
