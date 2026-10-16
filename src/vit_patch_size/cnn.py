"""A small CNN, sized to match the ViT's parameter count.

Supplementary reference only: the graded experiment is ViT 4x4 versus 8x8. The
CNN exists so the report can say how the transformers sit against a model with
built-in locality and weight sharing when everything else is equal - the same
split, normalization, optimizer, learning rate, batch size and batch order,
dropout rate, and no augmentation for any model.

Three 3x3 conv + BatchNorm + ReLU blocks with 2x2 max pooling after the first
two, global average pooling, dropout, and a linear head to 10 logits. Widths
(40, 64, 86) give 75,096 parameters against the 4x4 ViT's 75,082.
"""

import torch
from torch import nn

WIDTHS = (40, 64, 86)


class SmallCNN(nn.Module):
    def __init__(self, widths=WIDTHS, dropout: float = 0.1, num_classes: int = 10):
        super().__init__()
        c1, c2, c3 = widths

        def block(i, o, pool):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            return nn.Sequential(*layers, nn.MaxPool2d(2)) if pool else nn.Sequential(*layers)

        self.features = nn.Sequential(block(3, c1, True), block(c1, c2, True),
                                      block(c2, c3, False))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                  # (B, c3, 8, 8)
        x = x.mean(dim=(2, 3))                # global average pool -> (B, c3)
        return self.head(self.drop(x))        # raw logits (B, 10)
