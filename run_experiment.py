"""Run the graded experiments: 4x4 and 8x8 patches, nothing else changed.

Usage
-----
    .venv/bin/python run_experiment.py --run both
    .venv/bin/python run_experiment.py --run p4
    .venv/bin/python run_experiment.py --run both --smoke    # tiny, to results/smoke/

Writes ``results/p4.json`` and ``results/p8.json`` (schema: README.md, "Run-file schema")
and the figures in ``plots/``. ``--smoke`` shrinks the data and epochs and
writes under ``results/smoke`` and ``plots/smoke`` so a development run can
never be mistaken for the real one.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vit_patch_size import RunConfig, data, plots, train  # noqa: E402
from vit_patch_size._config import PLOTS_DIR, RESULTS_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", choices=("p4", "p8", "both"), default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    base = RunConfig(device=args.device)
    results_dir, plots_dir = RESULTS_DIR, PLOTS_DIR
    if args.smoke:
        base = RunConfig(device=args.device, epochs=1)
        results_dir, plots_dir = RESULTS_DIR / "smoke", PLOTS_DIR / "smoke"

    split = data.load_split(base)
    if args.smoke:
        n = 1000
        for k in ("x_train", "y_train"):
            split[k] = split[k][:n]
        split["raw_train"] = split["raw_train"][:n]

    configs = {"p4": base, "p8": base.with_patch(8)}
    wanted = ["p4", "p8"] if args.run == "both" else [args.run]
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for name in wanted:
        result = train.run(configs[name], split)
        (results_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        runs[name] = result
        plots.learning_curves(result, name, plots_dir / f"{name}_curves.png")
        plots.predictions_panel(result, split["raw_val"], plots_dir / f"{name}_predictions.png")
        print(f"[{name}] final val acc {result['final_val_acc']:.4f}, "
              f"{result['total_seconds']:.1f}s -> results/{'smoke/' if args.smoke else ''}{name}.json")

    plots.sample_grid(split["raw_train"], split["y_train"], plots_dir / "samples.png")
    if len(runs) == 2:
        plots.comparison_curves(runs, plots_dir / "comparison_curves.png")
        plots.attention_cost(runs, plots_dir / "attention_cost.png")
        plots.per_class_accuracy(runs, plots_dir / "per_class_accuracy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
