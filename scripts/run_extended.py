"""Extended training: both patch sizes and a parameter-matched CNN, to convergence.

The graded comparison stops at five epochs, where both ViTs are still
improving. This trains until validation loss has not improved for ``patience``
epochs (or ``max_epochs``), for three arms - ViT 4x4, ViT 8x8, a CNN - over
several seeds, on the same split, optimizer, batch size and batch order.
Everything is recorded per epoch, so accuracy at fixed optimizer-step budgets
can be read off afterwards. Validation only; the test split is never loaded.

It resumes: a finished (arm, seed) pair already in the output file is skipped,
unless ``--fresh`` is given (``run_all.py`` passes it, so the one-command
reproduction really retrains).

Usage
-----
    .venv/bin/python scripts/run_extended.py [--seeds 42 43 44 45 46]
        [--max-epochs 150] [--patience 10] [--arms p4 p8 cnn] [--out FILE] [--fresh]
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vit_patch_size import RunConfig, data, extended, plots  # noqa: E402
from vit_patch_size._config import PLOTS_DIR, RESULTS_DIR  # noqa: E402

DEFAULT_OUT = RESULTS_DIR / "supplementary" / "extended.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--arms", nargs="+", default=["p4", "p8", "cnn"])
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing output and retrain every run")
    args = ap.parse_args()

    base = RunConfig()
    split = data.load_split(base)          # fixed at seed 42, like every other run
    args.out.parent.mkdir(parents=True, exist_ok=True)

    payload = {"max_epochs": args.max_epochs, "patience": args.patience,
               "stop_rule": "validation loss has not improved for `patience` epochs",
               "split_checksum": split["manifest"]["checksum"],
               "n_val": base.n_val, "batch_size": base.batch_size,
               "steps_per_epoch": -(-base.n_train // base.batch_size), "runs": []}
    if args.out.is_file() and not args.fresh:
        old = json.loads(args.out.read_text())
        if (old["max_epochs"], old["patience"], old["split_checksum"]) == (
                args.max_epochs, args.patience, payload["split_checksum"]):
            payload["runs"] = old["runs"]
        else:
            raise SystemExit(f"{args.out.name} was made with different settings; "
                             f"move it aside or pass --out.")
    done = {(r["arm"], r["seed"]) for r in payload["runs"]}

    for seed in args.seeds:
        for arm in args.arms:
            if (arm, seed) in done:
                continue
            cfg = replace(base, seed=seed)
            payload["runs"].append(extended.run(arm, cfg, split, args.max_epochs, args.patience))
            tmp = args.out.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload) + "\n")
            tmp.replace(args.out)          # atomic: a stop mid-run never corrupts it

    if args.out == DEFAULT_OUT:
        plots.extended_curves(payload, PLOTS_DIR / "extended_curves.png")
    print(f"Wrote {args.out} ({len(payload['runs'])} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
