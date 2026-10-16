"""What machine and libraries produced a run, recorded into the run file."""

import platform
import subprocess
import sys

import numpy
import torch
import torchvision


def hardware(device) -> dict:
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True).stdout.strip() or chip
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "chip": chip,
        "platform": platform.platform(),
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def software() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
    }


def synchronize(device) -> None:
    """Wait for queued GPU work so a wall-clock reading is the work's cost."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
