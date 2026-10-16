"""A small Vision Transformer on CIFAR-10 for a controlled patch-size comparison."""

from ._config import CLASSES, COMPARED_FIELDS, RunConfig, resolve_device

__all__ = ["CLASSES", "COMPARED_FIELDS", "RunConfig", "resolve_device"]
