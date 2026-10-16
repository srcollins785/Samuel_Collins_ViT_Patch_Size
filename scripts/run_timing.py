"""Why does a 14.6x smaller attention map not give a 14.6x faster run, or 14.6x less memory?

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

import argparse
import gc
import json
import statistics
import subprocess
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


def memory_worker(patch: int) -> None:
    """Runs in its own process so the allocator holds nothing from another model.

    Measures the memory the autograd graph keeps alive after one forward pass
    (the activations saved for backward) and, before it, what parameters, Adam
    state and the input batch occupy. MPS only: it is the one backend here with
    a queryable allocator. Prints one JSON line.
    """
    cfg = RunConfig().with_patch(patch)
    device = torch.device("mps")
    model = build_model(cfg).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    x = torch.randn(cfg.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (cfg.batch_size,), device=device)
    for _ in range(3):                      # allocate Adam state, settle the allocator
        opt.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(x), y).backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.mps.synchronize()
    base = torch.mps.current_allocated_memory()
    out = model(x)                           # keep the graph alive
    torch.mps.synchronize()
    saved = torch.mps.current_allocated_memory() - base
    del out
    length = cfg.tokens
    print(json.dumps({
        "patch_size": patch, "tokens": length,
        "static_mb": base / 1e6, "saved_activations_mb": saved / 1e6,
        # one (B, heads, L, L) float32 score or weight tensor
        "one_attention_map_mb": cfg.batch_size * cfg.heads * length * length * 4 / 1e6,
    }))


def measure_memory() -> list:
    rows = []
    for patch in (4, 8):
        done = subprocess.run([sys.executable, __file__, "--memory-worker", str(patch)],
                              capture_output=True, text=True, check=True)
        rows.append(json.loads(done.stdout.strip().splitlines()[-1]))
        r = rows[-1]
        print(f"mps P={patch}: parameters+Adam+input {r['static_mb']:.2f} MB, "
              f"saved activations {r['saved_activations_mb']:.1f} MB, "
              f"one attention map {r['one_attention_map_mb']:.2f} MB")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memory-worker", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.memory_worker:
        memory_worker(args.memory_worker)
        return 0

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    rows = []
    for d in devices:
        for p in (4, 8):
            r = measure(p, d)
            rows.append(r)
            print(f"{d:>3} P={p}: step {r['train_step_ms']:.2f} ms, forward {r['forward_ms']:.2f} ms, "
                  f"scores+softmax {r['score_and_softmax_ms']:.3f} ms")
    memory = measure_memory() if torch.backends.mps.is_available() else []
    out = RESULTS_DIR / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing.json").write_text(json.dumps(
        {"warmup": WARMUP, "steps": STEPS, "batch_size": RunConfig().batch_size,
         "threads": torch.get_num_threads(), "rows": rows, "memory": memory},
        indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
