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
        "val_labels": [0, 1, 2, 3],
        "val_preds": [0, 1, 0, 0] if patch == 4 else [0, 0, 2, 0],
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
    assert rd.seed_stats(data) is None  # no replication: the report must say so


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


def test_seed_stats_and_agreement(results):
    (results / "supplementary").mkdir()
    (results / "supplementary" / "seed_replication.json").write_text(json.dumps({
        "n_val": 2000,
        "by_run": {
            "p4": {"seeds": [42, 43, 44], "val_acc": [0.45, 0.40, 0.42], "seconds": [10, 10, 10]},
            "p8": {"seeds": [42, 43, 44], "val_acc": [0.42, 0.41, 0.39], "seconds": [4, 4, 4]},
        }}))
    data = rd.load(results)
    s = rd.seed_stats(data)
    assert s["n"] == 3 and s["p4_wins"] == 2
    assert s["diffs"] == pytest.approx([0.03, -0.01, 0.03])
    assert s["diff_mean"] == pytest.approx(0.05 / 3)
    assert rd.graded_pair_in_replication(data) is True    # seed 42 = graded 0.45 / 0.42
    rep = json.loads((results / "supplementary" / "seed_replication.json").read_text())
    rep["by_run"]["p4"]["val_acc"][0] = 0.46              # a replication that disagrees
    (results / "supplementary" / "seed_replication.json").write_text(json.dumps(rep))
    assert rd.graded_pair_in_replication(rd.load(results)) is False
    a = rd.agreement(data)
    assert sum(a[k] for k in ("both_right", "only_p4_right", "only_p8_right", "both_wrong")) == a["n"]


def make_ext(results, p4_acc=0.50, p8_acc=0.40, cnn_acc=0.60, epochs=30, graded=None,
             patience=2):
    """A tiny extended.json: three arms, two seeds, linear learning curves that peak
    at epoch 10. ``graded`` (the run dicts) makes the seed-42 ViT runs reproduce the
    graded first epochs, as the real ones do."""
    params = {"p4": 75_082, "p8": 81_226, "cnn": 75_096}
    runs = []
    for seed in (42, 43):
        for arm, top in (("p4", p4_acc), ("p8", p8_acc), ("cnn", cnn_acc)):
            hist = [{"epoch": e, "train_loss": 2.0 - 0.01 * e, "train_acc": top,
                     "val_loss": 2.0 - 0.1 * e if e <= 10 else 1.0 + 0.1 * (e - 10),
                     "val_acc": top * e / 10 if e <= 10 else top * 0.9}
                    for e in range(1, epochs + 1)]
            if graded and seed == 42 and arm in graded:
                for g, h in zip(graded[arm]["history"], hist):
                    h["val_acc"], h["train_loss"] = g["val_acc"], g["train_loss"]
            hist = hist[:10 + patience]            # stop `patience` epochs after the best
            runs.append({"arm": arm, "seed": seed, "parameters": params[arm],
                         "epochs_run": len(hist), "stopped_early": True, "best_epoch": 10,
                         "best_val_loss": 1.0, "val_acc_at_best": hist[9]["val_acc"],
                         "final_val_acc": hist[-1]["val_acc"], "seconds": 1.0, "history": hist})
    (results / "supplementary").mkdir(exist_ok=True)
    (results / "supplementary" / "extended.json").write_text(json.dumps({
        "max_epochs": 150, "patience": patience, "split_checksum": "abc", "n_val": 2000,
        "batch_size": 64, "steps_per_epoch": 157, "runs": runs}))


def test_extended_queries(results):
    make_ext(results, epochs=40, patience=30)     # long enough for the 30-epoch budget
    data = rd.load(results)
    st = rd.ext_stats(data)
    assert st["p4"]["acc_mean"] == pytest.approx(0.50)
    assert st["cnn"]["best_epoch_mean"] == 10
    assert st["p4"]["at_budget"][5] == pytest.approx(0.25)   # linear ramp: 0.5 * 5/10
    assert st["p4"]["at_budget"][30] == pytest.approx(0.45)  # 0.9 * 0.5 after the peak
    g = rd.ext_gap_summary(data, "p4", "p8")
    assert g["mean"] == pytest.approx(0.10) and g["wins"] == g["n"] == 2
    at5 = rd.ext_gap_summary(data, "p4", "p8", 5)
    assert at5["mean"] == pytest.approx(0.05)
    assert rd.ext_gap_summary(data, "cnn", "p4", 5)["mean"] == pytest.approx(0.05)


def test_extended_absent_is_none_and_mismatched_seeds_are_rejected(results):
    assert rd.ext_stats(rd.load(results)) is None
    make_ext(results, epochs=40, patience=30)
    path = results / "supplementary" / "extended.json"
    d = json.loads(path.read_text())
    d["runs"] = [r for r in d["runs"] if not (r["arm"] == "cnn" and r["seed"] == 43)]
    path.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="different seeds"):
        rd.ext_diffs(rd.load(results), "cnn", "p4")
