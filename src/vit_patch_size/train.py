"""Train one ViT from scratch and return everything the report needs.

The return value is the run file's content (schema: README.md, "Run-file schema").
Nothing here writes to disk; ``run_experiment.py`` does, so the training
code can be tested without touching ``results/``.
"""

import random
import time

import numpy as np
import torch
from torch import nn

from . import data, env
from ._config import CLASSES, RunConfig, resolve_device
from .evaluate import evaluate
from .model import ViT, attention_entries, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config: RunConfig) -> ViT:
    """Initialize on the CPU under the seed, so initial weights do not depend
    on which device the run later trains on."""
    seed_everything(config.seed)
    return ViT(config.patch_size, config.embed_dim, config.depth, config.heads,
               config.mlp_dim, config.dropout)


def train_epoch(model, loader, optimizer, criterion, device) -> tuple:
    """One pass in train mode. Returns (mean loss, accuracy) over the pass,
    measured with dropout active as the weights were changing."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)   # raw logits: no softmax before the loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


def run(config: RunConfig, split: dict, log=print) -> dict:
    device = resolve_device(config.device)
    model = build_model(config)
    shapes = model.trace_shapes()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 betas=config.betas,
                                 weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = data.make_loader(split["x_train"], split["y_train"],
                              config.batch_size, config.seed)

    manifest = split["manifest"]
    entries = attention_entries(config.tokens, config.heads, config.depth)
    parameters = count_parameters(model)
    log(f"[{config.name}] P={config.patch_size} N={config.num_patches} "
        f"L={config.tokens} params={parameters:,} device={device}")

    history, total = [], 0.0
    for epoch in range(1, config.epochs + 1):
        env.synchronize(device)
        start = time.perf_counter()
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion,
                                            device)
        val = evaluate(model, split["x_val"], split["y_val"], device)
        env.synchronize(device)
        seconds = time.perf_counter() - start
        total += seconds
        # Not timed: the training set scored with dropout off, so the
        # train/validation gap compares like with like.
        clean = evaluate(model, split["x_train"], split["y_train"], device)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val["loss"], "val_acc": val["acc"], "seconds": seconds,
            "train_eval_loss": clean["loss"], "train_eval_acc": clean["acc"],
        })
        log(f"[{config.name}] epoch {epoch}/{config.epochs} "
            f"train {train_loss:.4f}/{train_acc:.4f}  "
            f"val {val['loss']:.4f}/{val['acc']:.4f}  {seconds:.1f}s")

    final = evaluate(model, split["x_val"], split["y_val"], device)
    y_val = split["y_val"].tolist()
    positions = data.fixed_example_positions(
        config.seed, config.n_val, config.n_fixed_examples, y_val)
    fixed = [{
        "position": pos, "index": int(split["val_indices"][pos]),
        "true": CLASSES[y_val[pos]], "pred": CLASSES[final["preds"][pos]],
        "prob": final["confidence"][pos],
    } for pos in positions]

    return {
        "config": {
            **config.to_dict(),
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss on raw logits",
            "split_checksum": manifest["checksum"],
            "normalization": manifest["normalization"],
            "normalization_mean": manifest["mean"],
            "normalization_std": manifest["std"],
            "batches_per_epoch": len(loader),
        },
        "model": {
            "patch_size": config.patch_size,
            "num_patches": config.num_patches,
            "tokens": config.tokens,
            "attn_entries": entries["per_head_per_layer"],
            "attn_entries_total": entries["total"],
            "parameters": parameters,
            "parameter_breakdown": model.parameter_breakdown(),
            "shapes": {k: list(v) for k, v in shapes.items()},
        },
        "history": history,
        "total_seconds": total,
        "final_val_acc": final["acc"],
        "final_val_loss": final["loss"],
        "fixed_examples": fixed,
        "val_preds": final["preds"],
        "val_labels": y_val,
        "hardware": env.hardware(device),
        "software": env.software(),
    }
