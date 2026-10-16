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

    ext = data.get("extended")
    if ext is None:
        problems.append("extended runs not found (scripts/run_extended.py)")
    else:
        arms = {k: rd.ext_runs(data, k) for k in rd.ARMS}
        seeds = {k: [r["seed"] for r in rs] for k, rs in arms.items()}
        if len({tuple(v) for v in seeds.values()}) != 1 or not seeds["p4"]:
            problems.append(f"extended: arms were not run on the same seeds: {seeds}")
        if ext["split_checksum"] != ca["split_checksum"]:
            problems.append("extended runs used a different split from the graded runs")
        vit4 = runs["p4"]["model"]["parameters"]
        for r in arms["cnn"]:
            if abs(r["parameters"] - vit4) / vit4 > 0.001:
                problems.append(f"extended: CNN has {r['parameters']} parameters, not within 0.1% of the 4x4 ViT's {vit4}")
        for arm, rs in arms.items():
            for r in rs:
                h = r["history"]
                losses = [x["val_loss"] for x in h]
                tag = f"extended {arm} seed {r['seed']}"
                if r["epochs_run"] != len(h):
                    problems.append(f"{tag}: epochs_run disagrees with the history length")
                if r["best_epoch"] != losses.index(min(losses)) + 1:
                    problems.append(f"{tag}: best_epoch is not the minimum-validation-loss epoch")
                if abs(r["val_acc_at_best"] - h[r["best_epoch"] - 1]["val_acc"]) > 1e-12:
                    problems.append(f"{tag}: val_acc_at_best disagrees with the history")
                if r["stopped_early"] and len(h) - r["best_epoch"] != ext["patience"]:
                    problems.append(f"{tag}: stopped early but not exactly `patience` epochs after the best")
                if not r["stopped_early"] and len(h) != ext["max_epochs"]:
                    problems.append(f"{tag}: not stopped early and not at the epoch cap")
        if rd.ext_matches_graded(data) is not True:
            problems.append("extended seed-42 ViT runs do not reproduce the graded first five epochs")
    if rd.graded_pair_in_replication(data) is False:
        problems.append("the seed-42 replication does not reproduce the graded runs")
    if not rd.same_fixed_examples(data):
        problems.append("the two runs show different fixed validation examples")
    if a["val_labels"] != b["val_labels"]:
        problems.append("the two runs were validated on different labels")
    return problems


def audit_report(report_md: Path, results_dir: Path) -> list:
    """Checks on the generated report: analysis length, and that the headline
    numbers it prints are the ones in the run files."""
    from generate_report import ANALYSIS_HEADING, analysis_words

    problems = []
    if not report_md.is_file():
        return [f"report not found: {report_md}"]
    text = report_md.read_text()
    if ANALYSIS_HEADING not in text:
        return ["report has no analysis section"]
    words = analysis_words(text)
    if not 400 <= words <= 600:
        problems.append(f"analysis is {words} words; the brief asks for 400-600")
    standalone = report_md.with_name("ViT_Patch_Size_Analysis.md")
    if not standalone.is_file():
        problems.append(f"standalone analysis not found: {standalone.name}")
    else:
        other = standalone.read_text()

        def paragraphs(t, heading):
            """The analysis text after its heading, up to the next rule or section."""
            if heading not in t:
                return None
            start = t.index(heading) + len(heading)
            stops = [x for x in (t.find("\n---", start), t.find("\n## ", start)) if x != -1]
            return t[start:min(stops) if stops else None].strip()

        mine, full = paragraphs(other, "## Analysis"), paragraphs(text, ANALYSIS_HEADING)
        if mine is None:
            problems.append("standalone analysis has no analysis section")
        elif mine != full:
            problems.append("standalone analysis differs from section 5 of the report")
    data = rd.load(results_dir)
    for name in rd.RUNS:
        acc = rd.pct(rd.final(data, name, "val_acc"))
        if acc not in text:
            problems.append(f"report does not contain {name} validation accuracy {acc}")
        params = f"{data['runs'][name]['model']['parameters']:,}"
        if params not in text:
            problems.append(f"report does not contain {name} parameter count {params}")
    for figure in sorted(set(__import__("re").findall(r"\(\.\./(plots/[^)]+)\)", text))):
        if not (REPO_ROOT / figure).is_file():
            problems.append(f"report references a missing figure: {figure}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--split", default=str(REPO_ROOT / "data" / "split_indices.json"))
    ap.add_argument("--report", action="store_true",
                    help="also audit report/ViT_Patch_Size_Report.md")
    args = ap.parse_args()
    problems = audit(Path(args.results_dir), Path(args.split))
    if args.report:
        problems += audit_report(REPO_ROOT / "report" / "ViT_Patch_Size_Report.md",
                                 Path(args.results_dir))
    if problems:
        print(f"AUDIT FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Audit passed: the comparison is controlled and the saved quantities agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
