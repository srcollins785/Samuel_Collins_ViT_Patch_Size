"""Package the submission ZIP from the git-tracked files.

Built with ``git archive`` rather than by walking the directory, so what
ships is exactly what is committed. That rules out the two failure modes
that matter here: a stray 64 MB ``data/CIFAR-10/`` or a virtual environment
riding along, and - worse - a figure or result that exists on this machine
but was never committed, which would make the ZIP unreproducible from the
repository.

The brief asks for the executed notebook, the analysis as a PDF, small
supporting files, setup and run instructions, and sources for adapted
code. All of those are tracked; the dataset and the checkpoints are not.

Usage
-----
    .venv/bin/python scripts/build_submission.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT.parent / "Samuel_Collins_ViT_Patch_Size.zip"
PREFIX = "Samuel_Collins_ViT_Patch_Size/"

REQUIRED = [
    "ViT_Patch_Size.ipynb",
    "report/ViT_Patch_Size_Report.pdf",
    "report/ViT_Patch_Size_Analysis.pdf",
    "README.md",
    "requirements.txt",
    "configuration.yaml",
    "data/split_indices.json",
    "results/p4.json",
    "results/p8.json",
]
FORBIDDEN_PREFIXES = ("data/cifar-10-batches-py", ".venv", "checkpoints/", "results/smoke", "plots/smoke")
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".gz")
SIZE_LIMIT_MB = 50


def main() -> int:
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("Uncommitted changes; the ZIP would not match the repository:\n")
        print(dirty)
        print("\nCommit first, or accept that the ZIP ships the committed "
              "version rather than the working tree.", file=sys.stderr)
        return 1

    result = subprocess.run(
        ["git", "archive", "--format=zip", "-9", f"--prefix={PREFIX}", "HEAD"],
        cwd=REPO_ROOT, capture_output=True,
    )
    if result.returncode != 0:
        print(result.stderr.decode()[-2000:], file=sys.stderr)
        return result.returncode
    OUTPUT.write_bytes(result.stdout)

    with zipfile.ZipFile(OUTPUT) as archive:
        names = [n[len(PREFIX):] for n in archive.namelist()
                 if n.startswith(PREFIX)]

    failures = []
    for required in REQUIRED:
        if required not in names:
            failures.append(f"missing required file {required}")
    for name in names:
        if name.startswith(FORBIDDEN_PREFIXES):
            failures.append(f"should not ship: {name}")
        if name.endswith(FORBIDDEN_SUFFIXES):
            failures.append(f"should not ship: {name}")

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    if size_mb > SIZE_LIMIT_MB:
        failures.append(f"{size_mb:.1f} MB exceeds the {SIZE_LIMIT_MB} MB "
                        f"sanity limit; something large got in")

    print(f"{OUTPUT.name}  -  {len(names)} files, {size_mb:.1f} MB")
    for required in REQUIRED:
        print(f"  present: {required}")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
