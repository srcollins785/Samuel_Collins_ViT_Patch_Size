"""Run the whole pipeline: tests, data, experiments, audit, report, notebook.

Each step is a separate script invoked as a subprocess and the first failure
stops the run, so any step can be run alone. Same orchestrator as Assignments
2 and 5. The notebook executes last because it imports ``src/`` and reads the
saved results: running it before the report would leave the notebook and the
PDF describing different builds.

Steps
-----
 1. pytest                      the suite must pass before results are believed
 2. build_cifar_split.py        the 10,000 / 2,000 split and its manifest
 3. run_experiment.py           the 4x4 and the 8x8 run
 4. run_supplementary.py        five-seed replication
 5. run_timing.py               per-step timing on CPU and GPU
 6. validate_results.py         the comparison is controlled; numbers agree
 7. write_configuration.py      configuration.yaml, from the run files
 8. generate_report.py          Markdown, from the run files
 9. build_report_pdf.py         PDF, via headless Chrome
10. validate_results.py --report  analysis length and figures
11. build_notebook.py           write and execute the notebook

Usage
-----
    .venv/bin/python scripts/run_all.py
    .venv/bin/python scripts/run_all.py --skip-training   # rebuild documents
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

TRAINING_STEPS = {"subset", "experiments", "supplementary", "timing"}

STEPS = [
    ("tests", [PYTHON, "-m", "pytest", "-q"]),
    ("subset", [PYTHON, "scripts/build_cifar_split.py"]),
    ("experiments", [PYTHON, "run_experiment.py", "--run", "both"]),
    ("supplementary", [PYTHON, "scripts/run_supplementary.py"]),
    ("timing", [PYTHON, "scripts/run_timing.py"]),
    ("audit", [PYTHON, "scripts/validate_results.py"]),
    ("configuration", [PYTHON, "scripts/write_configuration.py"]),
    ("report", [PYTHON, "scripts/generate_report.py"]),
    ("pdf", [PYTHON, "scripts/build_report_pdf.py"]),
    ("audit-report", [PYTHON, "scripts/validate_results.py", "--report"]),
    ("notebook", [PYTHON, "scripts/build_notebook.py", "--execute"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true",
                        help="rebuild the documents from existing results")
    parser.add_argument("--from-step", default=None,
                        help="start at this step name")
    args = parser.parse_args()

    steps = STEPS
    if args.from_step:
        names = [name for name, _ in steps]
        if args.from_step not in names:
            raise SystemExit(f"Unknown step {args.from_step!r}; "
                             f"expected one of {', '.join(names)}")
        steps = steps[names.index(args.from_step):]
    if args.skip_training:
        steps = [(n, c) for n, c in steps if n not in TRAINING_STEPS]

    started = time.perf_counter()
    for index, (name, command) in enumerate(steps, start=1):
        print(f"\n{'=' * 62}\n=== [{index}/{len(steps)}] {name}\n{'=' * 62}")
        result = subprocess.run(command, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"\nFAILED at step {name!r} (exit {result.returncode}). "
                  f"Nothing after it was run.", file=sys.stderr)
            return result.returncode

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 62}\nAll {len(steps)} steps passed in {elapsed:.1f}s.")
    print("Report:   report/ViT_Patch_Size_Report.pdf")
    print("Notebook: ViT_Patch_Size.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
