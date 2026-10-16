"""Audit the saved runs: the comparison must be controlled and the numbers coherent.

Fails (exit 1) with every problem listed, not just the first. Checks:

- both run files satisfy the schema (``report_data.load``)
- the two configs differ in ``patch_size`` (and ``name``) and in nothing else
- both recorded the same split checksum, which matches the committed manifest
- the test split was never used, and train/val indices are disjoint
- recorded shapes, token counts and L^2 entries agree with the patch size
- parameter breakdown sums to the parameter count and matches the hand calculation
- history arithmetic: epochs are 1..E, accuracies in [0, 1], total = sum of epochs
- the fixed examples are the same images in both runs
- final_val_acc equals the last history row and the saved predictions

Usage
-----
    .venv/bin/python scripts/validate_results.py [--results-dir results]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_data as rd  # noqa: E402

# Fields the comparison holds fixed. Mirrors vit_patch_size.COMPARED_FIELDS;
# duplicated on purpose so the audit, like the report, imports nothing from
# the package.
COMPARED = (
    "seed", "n_train", "n_val", "embed_dim", "depth", "heads", "mlp_dim",
    "dropout", "lr", "betas", "weight_decay", "batch_size", "epochs",
    "n_fixed_examples", "optimizer", "loss", "split_checksum", "normalization",
    "normalization_mean", "normalization_std",
)
ALLOWED_TO_DIFFER = {"name", "patch_size", "device", "batches_per_epoch"}
EXPECTED_PARAMS = {4: 75_082, 8: 81_226}


def audit(results_dir: Path, split_path: Path) -> list:
    problems = []
    data = rd.load(results_dir)
    runs = data["runs"]
    a, b = runs["p4"], runs["p8"]

    ca, cb = a["config"], b["config"]
    if (ca["patch_size"], cb["patch_size"]) != (4, 8):
        problems.append(f"patch sizes are {ca['patch_size']} and {cb['patch_size']}, expected 4 and 8")
    for key in sorted(set(ca) | set(cb)):
        if key in ALLOWED_TO_DIFFER:
            continue
        if ca.get(key) != cb.get(key):
            problems.append(f"config field {key!r} differs: {ca.get(key)!r} vs {cb.get(key)!r}")
    for key in COMPARED:
        if key not in ca or key not in cb:
            problems.append(f"config is missing compared field {key!r}")

    if split_path.is_file():
        manifest = json.loads(split_path.read_text())
        if ca["split_checksum"] != manifest["checksum"]:
            problems.append("run split checksum does not match data/split_indices.json")
        if manifest.get("test_split_used") is not False:
            problems.append("manifest does not state the test split was unused")
        if set(manifest["train_indices"]) & set(manifest["val_indices"]):
            problems.append("training and validation indices overlap")
        if (len(manifest["train_indices"]), len(manifest["val_indices"])) != (ca["n_train"], ca["n_val"]):
            problems.append("manifest sizes disagree with the run config")
    else:
        problems.append(f"split manifest not found: {split_path}")

    for name, run in runs.items():
        p = run["config"]["patch_size"]
        n = (32 // p) ** 2
        m = run["model"]
        if (m["num_patches"], m["tokens"], m["attn_entries"]) != (n, n + 1, (n + 1) ** 2):
            problems.append(f"{name}: token/attention counts disagree with patch size {p}")
        s = m["shapes"]
        if s["patches"][1:] != [n, 3 * p * p] or s["logits"][1:] != [10]:
            problems.append(f"{name}: recorded patch/logit shapes are wrong: {s}")
        if s["with_position_embedding"][1] != n + 1:
            problems.append(f"{name}: recorded token dimension is not N+1")
        if sum(m["parameter_breakdown"].values()) != m["parameters"]:
            problems.append(f"{name}: parameter breakdown does not sum to the total")
        if m["parameters"] != EXPECTED_PARAMS.get(p):
            problems.append(f"{name}: {m['parameters']} parameters, hand calculation says {EXPECTED_PARAMS.get(p)}")
        if m["attn_entries_total"] != m["attn_entries"] * run["config"]["heads"] * run["config"]["depth"]:
            problems.append(f"{name}: total attention entries inconsistent")

        h = run["history"]
        if [r["epoch"] for r in h] != list(range(1, run["config"]["epochs"] + 1)):
            problems.append(f"{name}: epochs are not 1..E")
        for r in h:
            for k in ("train_acc", "val_acc"):
                if not 0 <= r[k] <= 1:
                    problems.append(f"{name}: {k} out of range at epoch {r['epoch']}")
            if min(r["train_loss"], r["val_loss"]) <= 0:
                problems.append(f"{name}: non-positive loss at epoch {r['epoch']}")
        if abs(sum(r["seconds"] for r in h) - run["total_seconds"]) > 1e-6:
            problems.append(f"{name}: total_seconds is not the sum of epoch seconds")
        if abs(run["final_val_acc"] - h[-1]["val_acc"]) > 1e-9:
            problems.append(f"{name}: final_val_acc disagrees with the last history row")
        recomputed = sum(p_ == t for p_, t in zip(run["val_preds"], run["val_labels"])) / len(run["val_labels"])
        if abs(recomputed - run["final_val_acc"]) > 1e-9:
            problems.append(f"{name}: saved predictions give {recomputed:.4f}, run says {run['final_val_acc']:.4f}")
        if len(run["fixed_examples"]) != run["config"]["n_fixed_examples"]:
            problems.append(f"{name}: wrong number of fixed examples")

    if not rd.same_fixed_examples(data):
        problems.append("the two runs show different fixed validation examples")
    if a["val_labels"] != b["val_labels"]:
        problems.append("the two runs were validated on different labels")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--split", default=str(REPO_ROOT / "data" / "split_indices.json"))
    args = ap.parse_args()
    problems = audit(Path(args.results_dir), Path(args.split))
    if problems:
        print(f"AUDIT FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Audit passed: the comparison is controlled and the saved quantities agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
