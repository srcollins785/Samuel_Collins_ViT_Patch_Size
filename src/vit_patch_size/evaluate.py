"""Evaluation: dropout off, no gradient updates."""

import torch
from torch import nn


@torch.no_grad()
def evaluate(model: nn.Module, x, y, device, batch_size: int = 500) -> dict:
    """Loss, accuracy, predictions and class probabilities on (x, y).

    ``model.eval()`` disables dropout and ``torch.no_grad()`` stops gradient
    tracking, so this cannot change a parameter. The previous training mode is
    restored on exit.
    """
    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, preds, probs = 0.0, 0, [], []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item()
        p = logits.softmax(dim=-1)
        conf, pred = p.max(dim=-1)
        correct += (pred == yb).sum().item()
        preds.append(pred.cpu())
        probs.append(conf.cpu())
    model.train(was_training)
    return {
        "loss": total_loss / len(x),
        "acc": correct / len(x),
        "preds": torch.cat(preds).tolist(),
        "confidence": torch.cat(probs).tolist(),
    }
