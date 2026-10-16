"""Download CIFAR-10 (training split only) and record the train/val split.

The images are not committed - the brief forbids submitting dataset downloads
- so what goes into the repository is ``data/split_indices.json``: indices,
seed, per-class counts, normalization statistics and a checksum.

Usage
-----
    .venv/bin/python scripts/build_cifar_split.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vit_patch_size import RunConfig, data  # noqa: E402
from vit_patch_size._config import SPLIT_PATH  # noqa: E402


def main() -> int:
    config = RunConfig()
    m = data.write_split(config)
    print(f"Population     {m['population']:,} (official train split; test split not used)")
    print(f"Split          {m['n_train']:,} train / {m['n_val']:,} val at seed {m['seed']}")
    print(f"Per class      train {m['train_per_class']}  val {m['val_per_class']}")
    print(f"Mean / std     {[round(v, 4) for v in m['mean']]} / {[round(v, 4) for v in m['std']]}")
    print(f"Checksum       {m['checksum']}")
    print(f"Wrote          {SPLIT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
