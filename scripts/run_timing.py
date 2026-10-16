"""Why does a 14.6x smaller attention map not give a 14.6x faster run?

The brief asks for an explanation, so this measures instead of asserting one.
For each patch size and device it times one optimizer step (forward, backward,
update) at the graded batch size, after warm-up, as a median over many steps,
and splits out the forward pass of the attention score computation alone
(Q K^T and softmax) from the whole forward pass. Supplementary evidence:
the graded runtimes are the ones in results/p4.json and results/p8.json.

Usage
-----
    .venv/bin/python scripts/run_timing.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from vit_patch_size import RunConfig, env  # noqa: E402
from vit_patch_size._config import RESULTS_DIR  # noqa: E402
from vit_patch_size.train import build_model  # noqa: E402

WARMUP, STEPS = 20, 100


def median_ms(fn, device) -> float:
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(STEPS):
        env.synchronize(device)
        t = time.perf_counter()
        fn()
        env.synchronize(device)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def measure(patch: int, device_name: str) -> dict:
    cfg = RunConfig().with_patch(patch)
    device = torch.device(device_name)
    model = build_model(cfg).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(cfg.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (cfg.batch_size,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()

    def forward():
        with torch.no_grad():
            model(x)

    length, dh = cfg.tokens, cfg.embed_dim // cfg.heads
    q = torch.randn(cfg.batch_size, cfg.heads, length, dh, device=device)
    k = torch.randn_like(q)

    def scores():
        with torch.no_grad():
            (q @ k.transpose(-2, -1) / dh ** 0.5).softmax(-1)

    return {"patch_size": patch, "device": device_name, "tokens": length,
            "train_step_ms": median_ms(step, device),
            "forward_ms": median_ms(forward, device),
            "score_and_softmax_ms": median_ms(scores, device)}


def main() -> int:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    rows = []
    for d in devices:
        for p in (4, 8):
            r = measure(p, d)
            rows.append(r)
            print(f"{d:>3} P={p}: step {r['train_step_ms']:.2f} ms, forward {r['forward_ms']:.2f} ms, "
                  f"scores+softmax {r['score_and_softmax_ms']:.3f} ms")
    out = RESULTS_DIR / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing.json").write_text(json.dumps(
        {"warmup": WARMUP, "steps": STEPS, "batch_size": RunConfig().batch_size,
         "threads": torch.get_num_threads(), "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
