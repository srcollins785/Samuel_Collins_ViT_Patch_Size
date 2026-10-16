"""Load what the runs wrote, and answer questions about it.

The report is not allowed to state a number it did not read from disk, and it
is not allowed to import the package. This module is the boundary: it reads
``results/*.json`` and ``data/split_indices.json`` and exposes small query
functions, so the prose asks a question and gets the measured answer instead
of a remembered one. Same design as Assignment 5's ``report_data.py``.

The contract with the experiment code is the run-file schema in PLAN.md,
section 5. ``load`` checks it up front, so a malformed or half-written run
fails here with a clear message rather than as a KeyError inside a sentence.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

NA = "N/A"
RUNS = ("p4", "p8")

REQUIRED_RUN_KEYS = (
    "config", "model", "history", "total_seconds", "fixed_examples",
    "hardware", "software",
)
REQUIRED_CONFIG_KEYS = (
    "patch_size", "embed_dim", "depth", "heads", "mlp_dim", "dropout", "lr",
    "batch_size", "epochs", "seed", "n_train", "n_val", "split_checksum",
)
REQUIRED_HISTORY_KEYS = (
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds",
)


def load(results_dir: Path = RESULTS_DIR, runs=RUNS) -> dict:
    """Everything the report is built from. Raises SystemExit on a bad run."""
    data = {"runs": {}}
    for name in runs:
        path = results_dir / f"{name}.json"
        if not path.is_file():
            raise SystemExit(
                f"Missing {path.name}. Run the experiments first:\n"
                f"  .venv/bin/python run_experiment.py --run both"
            )
        run = json.loads(path.read_text())
        _check(name, run)
        data["runs"][name] = run

    seeds = results_dir / "supplementary" / "seed_replication.json"
    data["seeds"] = json.loads(seeds.read_text()) if seeds.is_file() else None
    return data


def _check(name: str, run: dict) -> None:
    missing = [k for k in REQUIRED_RUN_KEYS if k not in run]
    missing += [f"config.{k}" for k in REQUIRED_CONFIG_KEYS
                if k not in run.get("config", {})]
    for i, row in enumerate(run.get("history", [])):
        missing += [f"history[{i}].{k}" for k in REQUIRED_HISTORY_KEYS
                    if k not in row]
    if missing:
        raise SystemExit(f"{name}.json is missing: {', '.join(missing)}")
    if len(run["history"]) != run["config"]["epochs"]:
        raise SystemExit(
            f"{name}.json has {len(run['history'])} history rows for "
            f"{run['config']['epochs']} configured epochs; the run is incomplete."
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(number, places: int = 4, default: str = NA) -> str:
    return default if number is None else f"{number:.{places}f}"


def pct(value, places: int = 1, default: str = NA) -> str:
    return default if value is None else f"{100 * value:.{places}f}%"


def seconds(value, default: str = NA) -> str:
    if value is None:
        return default
    if value < 60:
        return f"{value:.1f} s"
    minutes, remainder = divmod(value, 60)
    return f"{int(minutes)} m {remainder:.0f} s"


def figure(path: str, caption: str) -> str:
    """A figure reference relative to report/, where the Markdown lives."""
    return f"![{caption}](../{path})\n\n*{caption}*\n"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def final(data: dict, run: str, field: str):
    history = data["runs"][run]["history"]
    return history[-1][field] if history else None


def best(data: dict, run: str, field: str = "val_acc") -> dict:
    return max(data["runs"][run]["history"], key=lambda r: r[field])


def config(data: dict, run: str, field: str):
    return data["runs"][run]["config"][field]


def model(data: dict, run: str, field: str):
    return data["runs"][run]["model"][field]


def tokens(patch_size: int, image_size: int = 32) -> tuple:
    """(N patch tokens, L = N + 1 total tokens, L**2 attention entries)."""
    n = (image_size // patch_size) ** 2
    return n, n + 1, (n + 1) ** 2


def attention_entry_ratio(data: dict) -> float:
    """How many times more L**2 score entries p4 computes than p8."""
    return model(data, "p4", "attn_entries") / model(data, "p8", "attn_entries")


def runtime_ratio(data: dict) -> float:
    return data["runs"]["p4"]["total_seconds"] / data["runs"]["p8"]["total_seconds"]


def higher_final_val_acc(data: dict) -> str:
    """Which run ended with the higher validation accuracy. Measured."""
    return max(RUNS, key=lambda r: final(data, r, "val_acc"))


def val_acc_gap(data: dict) -> float:
    a, b = (final(data, r, "val_acc") for r in RUNS)
    return abs(a - b)


def gap_exceeds_seed_noise(data: dict) -> "bool | None":
    """None when no seed replication exists - the report must then say so."""
    seeds = data.get("seeds")
    if not seeds:
        return None
    spread = max(
        max(s["val_acc"]) - min(s["val_acc"]) for s in seeds["by_run"].values()
    )
    return val_acc_gap(data) > spread


def per_class_errors(run: dict) -> dict:
    """{true_label: count of fixed examples predicted wrongly}."""
    errors: dict = {}
    for ex in run["fixed_examples"]:
        if ex["true"] != ex["pred"]:
            errors[ex["true"]] = errors.get(ex["true"], 0) + 1
    return errors


def same_fixed_examples(data: dict) -> bool:
    """Both runs must show the same eight validation images."""
    a, b = (
        [ex["index"] for ex in data["runs"][r]["fixed_examples"]] for r in RUNS
    )
    return a == b
