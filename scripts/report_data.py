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
    timing = results_dir / "supplementary" / "timing.json"
    data["timing"] = json.loads(timing.read_text()) if timing.is_file() else None
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


def seed_stats(data: dict) -> "dict | None":
    """Per-seed accuracies, paired differences (p4 minus p8) and their summary.

    None when no replication exists - the report must then say so. The
    comparison is paired: for each seed both patch sizes share the split and
    differ only in patch size, so the per-seed difference is the quantity
    whose spread says whether the graded gap is signal.
    """
    seeds = data.get("seeds")
    if not seeds:
        return None
    p4, p8 = seeds["by_run"]["p4"], seeds["by_run"]["p8"]
    if p4["seeds"] != p8["seeds"]:
        raise SystemExit("seed replication: the two runs used different seeds")
    diffs = [a - b for a, b in zip(p4["val_acc"], p8["val_acc"])]
    n = len(diffs)

    def mean(xs):
        return sum(xs) / len(xs)

    def sd(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    return {
        "n": n,
        "seeds": p4["seeds"],
        "p4": p4["val_acc"], "p8": p8["val_acc"],
        "p4_mean": mean(p4["val_acc"]), "p8_mean": mean(p8["val_acc"]),
        "p4_sd": sd(p4["val_acc"]), "p8_sd": sd(p8["val_acc"]),
        "diffs": diffs,
        "diff_mean": mean(diffs), "diff_sd": sd(diffs),
        "diff_min": min(diffs), "diff_max": max(diffs),
        "p4_wins": sum(d > 0 for d in diffs),
        "p4_seconds_mean": mean(seeds["by_run"]["p4"]["seconds"]),
        "p8_seconds_mean": mean(seeds["by_run"]["p8"]["seconds"]),
        # A standard error for one accuracy on n_val images, binomial.
        "binomial_se": (mean(p4["val_acc"] + p8["val_acc"])
                        * (1 - mean(p4["val_acc"] + p8["val_acc"]))
                        / seeds["n_val"]) ** 0.5,
    }


def graded_pair_in_replication(data: dict) -> "bool | None":
    """The seed-42 replication must reproduce the graded runs exactly."""
    stats = seed_stats(data)
    if stats is None or 42 not in stats["seeds"]:
        return None
    i = stats["seeds"].index(42)
    return (stats["p4"][i] == final(data, "p4", "val_acc")
            and stats["p8"][i] == final(data, "p8", "val_acc"))


def generalization_gap(data: dict, run: str) -> float:
    """Clean training accuracy minus validation accuracy at the last epoch."""
    return final(data, run, "train_eval_acc") - final(data, run, "val_acc")


def agreement(data: dict) -> dict:
    """How often the two models agree on the validation set, and who is right
    when they disagree. Measured from the saved predictions."""
    a, b = data["runs"]["p4"], data["runs"]["p8"]
    labels = a["val_labels"]
    both = only4 = only8 = neither = 0
    for t, x, y in zip(labels, a["val_preds"], b["val_preds"]):
        if x == t and y == t:
            both += 1
        elif x == t:
            only4 += 1
        elif y == t:
            only8 += 1
        else:
            neither += 1
    n = len(labels)
    return {"both_right": both, "only_p4_right": only4, "only_p8_right": only8,
            "both_wrong": neither, "n": n}


def per_class_accuracy(run: dict) -> dict:
    """{class index: validation accuracy}, from the saved predictions."""
    right: dict = {}
    count: dict = {}
    for t, p in zip(run["val_labels"], run["val_preds"]):
        count[t] = count.get(t, 0) + 1
        right[t] = right.get(t, 0) + (t == p)
    return {c: right[c] / count[c] for c in sorted(count)}


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


CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")


def largest_class_gap(data: dict) -> dict:
    """The class where p4 and p8 differ most in validation accuracy (p4 - p8)."""
    a = per_class_accuracy(data["runs"]["p4"])
    b = per_class_accuracy(data["runs"]["p8"])
    c = max(a, key=lambda k: abs(a[k] - b[k]))
    per_class = data["runs"]["p4"]["val_labels"].count(c)
    return {"class": CLASS_NAMES[c], "p4": a[c], "p8": b[c],
            "images": round((a[c] - b[c]) * per_class), "per_class": per_class}


def timing_row(data: dict, device: str, patch: int) -> "dict | None":
    t = data.get("timing")
    if not t:
        return None
    for row in t["rows"]:
        if row["device"] == device and row["patch_size"] == patch:
            return row
    return None


def timing_ratio(data: dict, device: str, field: str = "train_step_ms") -> "float | None":
    a, b = timing_row(data, device, 4), timing_row(data, device, 8)
    return None if a is None or b is None else a[field] / b[field]


def swin_window_entries(grid: int = 56, window: int = 7) -> dict:
    """Attention entries for one layer over a grid x grid token map: global
    attention versus non-overlapping window x window attention. Pure arithmetic
    (Swin has no class token), used in the report's Swin discussion."""
    tokens = grid * grid
    windows = (grid // window) ** 2
    per_window = (window * window) ** 2
    return {"tokens": tokens, "global": tokens ** 2, "windows": windows,
            "per_window": per_window, "windowed": windows * per_window,
            "ratio": tokens ** 2 / (windows * per_window)}


def seconds_range(data: dict, run: str) -> tuple:
    """(min, max) wall-clock seconds across the seed replication for one run."""
    values = data["seeds"]["by_run"][run]["seconds"]
    return min(values), max(values)
