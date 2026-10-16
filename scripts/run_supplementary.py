"""Seed replication: how much does validation accuracy move from seed alone?

The graded comparison is one pair of runs at seed 42. With 2,000 validation
images a single accuracy has a binomial standard error near one percentage
point, so a gap of a few points between two single runs is not interpretable
by itself. This repeats both patch sizes at several seeds on the *same split*,
varying only what ``seed`` controls - the initial weights, the batch order and
the dropout masks - and records the final validation accuracy of each.

It is supplementary evidence: the graded results are results/p4.json and
results/p8.json, untouched by this script.

Usage
-----
    .venv/bin/python scripts/run_supplementary.py [--seeds 42 43 44 45 46]
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vit_patch_size import RunConfig, data, plots, train  # noqa: E402
from vit_patch_size._config import PLOTS_DIR, RESULTS_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    base = RunConfig()
    split = data.load_split(base)           # the split is fixed at seed 42
    quiet = lambda *_: None

    by_run = {"p4": {"seeds": [], "val_acc": [], "val_loss": [], "seconds": []},
              "p8": {"seeds": [], "val_acc": [], "val_loss": [], "seconds": []}}
    for seed in args.seeds:
        for name, patch in (("p4", 4), ("p8", 8)):
            cfg = replace(base.with_patch(patch), seed=seed)
            r = train.run(cfg, split, log=quiet)
            row = by_run[name]
            row["seeds"].append(seed)
            row["val_acc"].append(r["final_val_acc"])
            row["val_loss"].append(r["final_val_loss"])
            row["seconds"].append(r["total_seconds"])
            print(f"seed {seed} {name}: val acc {r['final_val_acc']:.4f}  {r['total_seconds']:.1f}s")

    out = RESULTS_DIR / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "what_varies": "initial weights, batch order and dropout masks (seed); split held fixed",
        "split_checksum": split["manifest"]["checksum"],
        "n_val": base.n_val,
        "by_run": by_run,
    }
    (out / "seed_replication.json").write_text(json.dumps(payload, indent=2) + "\n")
    plots.seed_replication(by_run, PLOTS_DIR / "seed_replication.png")
    print(f"Wrote {out.relative_to(REPO_ROOT)}/seed_replication.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
