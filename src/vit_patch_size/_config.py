"""Run configuration and the paths everything else reads and writes.

One frozen dataclass per experiment. The two experiments differ in exactly
one field, ``patch_size``, and ``COMPARED_FIELDS`` names every field that must
be identical between them. ``scripts/validate_results.py`` diffs the saved
configurations against that list, so adding a knob without deciding whether
it belongs to the controlled comparison is not possible.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
REPORT_DIR = REPO_ROOT / "report"
SPLIT_PATH = DATA_DIR / "split_indices.json"

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
IMAGE_SIZE = 32

# Everything the comparison holds fixed. ``patch_size`` is the one thing
# allowed to vary and ``name`` differs by construction. ``device`` is a
# property of the machine rather than of the experiment.
COMPARED_FIELDS = (
    "seed", "n_train", "n_val", "embed_dim", "depth", "heads", "mlp_dim",
    "dropout", "lr", "betas", "weight_decay", "batch_size", "epochs",
    "n_fixed_examples",
)


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines a run. Defaults are the 4x4 model."""

    name: str = "p4"
    patch_size: int = 4
    # Fixed by the brief.
    embed_dim: int = 64
    depth: int = 2
    heads: int = 4
    mlp_dim: int = 128
    epochs: int = 5
    n_train: int = 10_000
    n_val: int = 2_000
    # The brief's starting configuration, unchanged.
    lr: float = 3e-4
    batch_size: int = 64
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    # The brief says evaluation disables dropout, so training uses some.
    # 0.1 is the ViT paper's default for its small models.
    dropout: float = 0.1
    seed: int = 42
    n_fixed_examples: int = 8
    device: str = "auto"

    def with_patch(self, patch_size: int) -> "RunConfig":
        """The controlled comparison: this run with one field moved."""
        return replace(self, name=f"p{patch_size}", patch_size=patch_size)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["betas"] = list(data["betas"])
        return data

    @property
    def num_patches(self) -> int:
        return (IMAGE_SIZE // self.patch_size) ** 2

    @property
    def tokens(self) -> int:
        return self.num_patches + 1


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
