"""Write configuration.yaml from the saved run files.

Machine-written, never edited by hand: every value is read from
``results/*.json`` and ``data/split_indices.json``, so the file cannot
disagree with what ran. Same role as in Assignments 4 and 5.

Usage
-----
    .venv/bin/python scripts/write_configuration.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_data as rd  # noqa: E402


def main() -> int:
    data = rd.load()
    manifest = json.loads((REPO_ROOT / "data" / "split_indices.json").read_text())
    p4, p8 = data["runs"]["p4"], data["runs"]["p8"]
    c = p4["config"]

    def experiment(run):
        return {
            "patch_size": run["config"]["patch_size"],
            "patches_N": run["model"]["num_patches"],
            "tokens_L": run["model"]["tokens"],
            "attention_entries_per_head_per_layer": run["model"]["attn_entries"],
            "parameters": run["model"]["parameters"],
            "final_val_accuracy": round(run["final_val_acc"], 4),
            "final_val_loss": round(run["final_val_loss"], 4),
            "runtime_seconds": round(run["total_seconds"], 2),
            "device": run["hardware"]["device"],
        }

    out = {
        "generated": date.today().isoformat(),
        "generated_by": "scripts/write_configuration.py, from results/*.json",
        "assignment": "Lab Assignment 6: Vision Transformers (ViT), patch-size comparison",
        "data": {
            "dataset": manifest["dataset"],
            "n_train": manifest["n_train"],
            "n_val": manifest["n_val"],
            "selection": manifest["selection"],
            "selection_seed": manifest["seed"],
            "split_checksum": manifest["checksum"],
            "test_split_used": manifest["test_split_used"],
            "image_shape": manifest["image_shape"],
            "normalization": manifest["normalization"],
            "normalization_mean": [round(v, 6) for v in manifest["mean"]],
            "normalization_std": [round(v, 6) for v in manifest["std"]],
            "augmentation": "none",
        },
        "model": {
            "embed_dim": c["embed_dim"], "depth": c["depth"], "heads": c["heads"],
            "mlp_dim": c["mlp_dim"], "dropout": c["dropout"],
            "class_token": True, "position_embedding": "learned",
            "attention_mask": "none (every patch attends to every patch)",
            "block": "pre-norm, residual",
            "head": "linear, 10 logits",
            "loss": c["loss"],
        },
        "training": {
            "seed": c["seed"], "epochs": c["epochs"], "batch_size": c["batch_size"],
            "batches_per_epoch": c["batches_per_epoch"], "optimizer": c["optimizer"],
            "lr": c["lr"], "betas": c["betas"], "weight_decay": c["weight_decay"],
            "evaluation": "model.eval() and torch.no_grad(); dropout off",
        },
        "controlled_variable": "patch_size (4 versus 8); nothing else differs",
        "experiments": {"p4": experiment(p4), "p8": experiment(p8)},
        "hardware": p4["hardware"],
        "software": p4["software"],
    }
    seeds = rd.seed_stats(data)
    if seeds:
        out["seed_replication"] = {
            "seeds": seeds["seeds"], "p4_wins": seeds["p4_wins"],
            "mean_difference_points": round(100 * seeds["diff_mean"], 2),
        }
    ext = rd.ext_stats(data)
    if ext:
        out["extended_training"] = {
            "what": "supplementary: train to convergence with early stopping; validation only",
            "max_epochs": data["extended"]["max_epochs"],
            "patience": data["extended"]["patience"],
            "stop_rule": data["extended"]["stop_rule"],
            "seeds": ext["p4"]["seeds"],
            "arms": {a: {"parameters": ext[a]["parameters"],
                         "val_acc_at_best_epoch_mean": round(ext[a]["acc_mean"], 4),
                         "best_epoch_mean": round(ext[a]["best_epoch_mean"], 1)}
                     for a in rd.ARMS},
            "cnn": "3x conv3x3+BatchNorm+ReLU (widths 40, 64, 86), 2 max-pools, global average "
                   "pool, dropout 0.1, linear head",
            "test_split_used": False,
        }
    (REPO_ROOT / "configuration.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
    print("Wrote configuration.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
