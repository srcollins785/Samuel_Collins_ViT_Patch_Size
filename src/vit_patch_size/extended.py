"""Train to convergence with early stopping, recording every epoch.

Supplementary experiment (see ``scripts/run_extended.py``). The graded runs are
five epochs; this asks what happens if training continues until validation
loss stops improving, for the 4x4 ViT, the 8x8 ViT and a parameter-matched CNN.

Selection uses the validation set only, and the official test split is never
loaded. The accuracy at the stopped epoch is therefore slightly optimistic,
equally so for every arm, because the epoch was chosen on the same images it is
scored on. Accuracy at fixed epoch budgets (same optimizer steps for every arm)
involves no selection and is reported alongside it.
"""

import time

import torch
from torch import nn

from . import data, env
from ._config import RunConfig, resolve_device
from .cnn import SmallCNN
from .evaluate import evaluate
from .model import count_parameters
from .train import build_model, seed_everything, train_epoch


def build_arm(arm: str, config: RunConfig) -> nn.Module:
    """``p4`` / ``p8`` are the ViTs of the graded experiment; ``cnn`` the reference."""
    if arm in ("p4", "p8"):
        return build_model(config.with_patch(int(arm[1:])))
    if arm == "cnn":
        seed_everything(config.seed)
        return SmallCNN(dropout=config.dropout)
    raise ValueError(f"unknown arm {arm!r}")


def should_stop(val_losses: list, patience: int) -> bool:
    """True once ``patience`` epochs have passed without a new best validation loss."""
    if len(val_losses) <= patience:
        return False
    best_epoch = min(range(len(val_losses)), key=val_losses.__getitem__)
    return len(val_losses) - 1 - best_epoch >= patience


def run(arm: str, config: RunConfig, split: dict, max_epochs: int, patience: int,
        log=print) -> dict:
    device = resolve_device(config.device)
    model = build_arm(arm, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas,
                                 weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = data.make_loader(split["x_train"], split["y_train"],
                              config.batch_size, config.seed)

    history, val_losses, total = [], [], 0.0
    for epoch in range(1, max_epochs + 1):
        env.synchronize(device)
        start = time.perf_counter()
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, device)
        val = evaluate(model, split["x_val"], split["y_val"], device)
        env.synchronize(device)
        total += time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val["loss"], "val_acc": val["acc"]})
        val_losses.append(val["loss"])
        if should_stop(val_losses, patience):
            break

    best = min(range(len(history)), key=lambda i: history[i]["val_loss"])
    log(f"[{arm} seed {config.seed}] {len(history)} epochs, best epoch {best + 1}, "
        f"val acc there {history[best]['val_acc']:.4f}, {total:.0f}s")
    return {
        "arm": arm, "seed": config.seed, "parameters": count_parameters(model),
        "epochs_run": len(history), "stopped_early": len(history) < max_epochs,
        "best_epoch": best + 1, "best_val_loss": history[best]["val_loss"],
        "val_acc_at_best": history[best]["val_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "seconds": total, "history": history,
    }
