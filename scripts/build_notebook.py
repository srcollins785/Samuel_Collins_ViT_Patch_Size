"""Write ViT_Patch_Size.ipynb, then let nbconvert execute it.

The notebook is a thin wrapper over ``src/vit_patch_size``. It calls the same
``train.run`` that ``run_experiment.py`` calls, so there is one implementation
and the notebook is a view onto it. It retrains both models live (about ten
seconds on this machine) and then compares the result with the saved
``results/*.json``, so the executed outputs are evidence the saved results
reproduce, not a second set of numbers.

Usage
-----
    .venv/bin/python scripts/build_notebook.py             # write it
    .venv/bin/python scripts/build_notebook.py --execute   # write and run it
"""

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "ViT_Patch_Size.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


CELLS = [
    md("""
# Patch Size in a Small Vision Transformer

**Samuel Collins** - Ph.D. Student, Department of Cyber-Physical Systems,
Clark Atlanta University

Computer Vision, Lab Assignment 6. Instructor: Dr. Kishor Datta Gupta.

A two-block ViT trained from scratch on a 10,000-image CIFAR-10 subset, once
with 4x4 patches and once with 8x8 patches, everything else identical.

This notebook is a thin wrapper over the `vit_patch_size` package in `src/`.
It retrains both models live with the same function `run_experiment.py` uses,
then checks the outcome against the saved `results/*.json` that the report is
built from. The written analysis is `report/ViT_Patch_Size_Report.pdf`.
"""),

    md("## 1. Setup and configuration"),
    code("""
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import Image, display

from vit_patch_size import CLASSES, COMPARED_FIELDS, RunConfig, data, plots, resolve_device
from vit_patch_size import env, train
from vit_patch_size.model import ViT, attention_entries, count_parameters

base = RunConfig()
configs = {"p4": base, "p8": base.with_patch(8)}
device = resolve_device()

print("hardware:", env.hardware(device))
print("software:", env.software())
for f in COMPARED_FIELDS:
    print(f"  {f:<18} {getattr(base, f)}")
differing = [f.name for f in fields(base)
             if getattr(configs['p4'], f.name) != getattr(configs['p8'], f.name)]
print("fields that differ between the two runs:", differing)
"""),

    md("""
## 2. The data

A stratified 10,000-image training subset and a disjoint 2,000-image
validation subset, both from CIFAR-10's official *training* split. The test
split is never loaded. Indices, seed and a checksum are committed in
`data/split_indices.json`; the images are not.
"""),
    code("""
split = data.load_split(base)
m = split["manifest"]
print(f"train {len(split['y_train']):,}  val {len(split['y_val']):,}  seed {m['seed']}")
print(f"checksum {m['checksum'][:16]}...  test split used: {m['test_split_used']}")
print("train per class", m["train_per_class"])
print("val per class  ", m["val_per_class"])
print("channel mean", [round(v, 4) for v in m["mean"]], " std", [round(v, 4) for v in m["std"]])
print("input tensor", tuple(split["x_train"].shape), "dtype", split["x_train"].dtype)
"""),
    md("### Sixteen labeled examples"),
    code("""
out = Path("plots/notebook"); out.mkdir(parents=True, exist_ok=True)
plots.sample_grid(split["raw_train"], split["y_train"].tolist(), out / "samples.png")
display(Image(filename=out / "samples.png"))
"""),

    md("""
## 3. The model

Written out by hand in `src/vit_patch_size/model.py` rather than built from
`nn.TransformerEncoderLayer`, because the report needs the attention weights
themselves. No code was copied from another repository: the structure follows
Dosovitskiy et al. (ICLR 2021) and Vaswani et al. (NeurIPS 2017).
`nn.MultiheadAttention` is used only in `tests/test_model.py`, as an oracle
that the hand-written attention must match given the same weights.

Specification, as the brief fixes it: 4x4 (then 8x8) non-overlapping patches,
one shared linear projection to width 64, a learnable class token, learned
position embeddings, two pre-norm encoder blocks with layer normalization and
residual connections, four heads, feed-forward width 128, and a linear head
producing 10 logits.
"""),
    code("""
import inspect
from vit_patch_size import model as vit_model

for obj in (vit_model.patchify, vit_model.Attention, vit_model.Block, vit_model.ViT):
    print(inspect.getsource(obj))
"""),

    md("""
## 3.1 Tensor shapes

Shapes come from a real forward pass (`trace_shapes`), not from arithmetic.
With P x P patches on a 32 x 32 image, N = (32/P)^2 patch tokens and
L = N + 1 tokens including the class token.
"""),
    code("""
rows = []
for name, cfg in configs.items():
    model = train.build_model(cfg)
    trace = model.trace_shapes(batch_size=64)
    print(f"{cfg.patch_size}x{cfg.patch_size}: N = (32/{cfg.patch_size})^2 = {cfg.num_patches}, "
          f"L = N + 1 = {cfg.tokens}, parameters = {count_parameters(model):,}")
    for stage, shp in trace.items():
        print(f"    {stage:<26} {shp}")
    print("    parameter breakdown:", model.parameter_breakdown())
    print()
"""),

    md("""
## 3.2 How attention works

Each token `t` is multiplied by three learned matrices to give a **query**
`q = W_Q t`, a **key** `k = W_K t` and a **value** `v = W_V t`.

- **Queries and keys decide the weights.** The score of token *i* for token
  *j* is the dot product `q_i . k_j`, a similarity. Scores are divided by
  `sqrt(d_key)` for stable gradients and passed through a softmax along *j*,
  giving non-negative weights that sum to 1.
- **Values produce the output.** The output for token *i* is the
  weight-weighted sum of every token's value vector.
- Four heads repeat this in four 16-dimensional subspaces; an output
  projection mixes them.

**Why positional information is needed.** Attention weights depend only on
token content, so without positions the patches are an unordered set and
shuffling an image's patches cannot change the output. The learned position
embedding is what lets the model use where a patch is. The cell below shows
both halves of this on the real model.

**Why there is no causal mask.** Classification has no generation order.
Every patch should attend to every patch in both directions. A causal mask
(future positions set to minus infinity before the softmax) belongs to the
transformer decoder; a ViT is encoder-only. With one, the class token at
position 0 could see nothing but itself.
"""),

    md("""
### Queries, keys, values: one worked example

Three value vectors and the query-1 scores `[2, 4, 4]` from the lecture
example, with exact softmax. The lecture slide rounds the weights to
`[0, 0.5, 0.5]`; the second row divides scores by sqrt(3), as the model does.
"""),
    code("""
import math
values = torch.tensor([[0., 0., 0.], [1., 4., 0.], [1., 3., 1.5]])
scores = torch.tensor([2., 4., 4.])
for label, s in (("raw scores", scores), ("scores / sqrt(3)", scores / math.sqrt(3))):
    w = s.softmax(0)
    print(f"{label:<18} weights {[round(x, 4) for x in w.tolist()]}  ->  "
          f"sum_i w_i v_i = {[round(x, 3) for x in (w @ values).tolist()]}")
"""),

    md("""
### Attention is global and unmasked

Every token attends to every token; the class token (position 0) puts
non-zero weight on later positions, which a causal mask would forbid.
"""),
    code("""
model = train.build_model(configs["p8"]).eval()
x = split["x_val"][:4]
with torch.no_grad():
    logits, maps = model(x, return_attn=True)
w = maps[0]
print("attention tensor (batch, heads, L, L):", tuple(w.shape))
print("rows sum to 1:", bool(torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)))
print("weight above the diagonal (a causal mask would make this 0):", float(torch.triu(w, 1).sum()))
print("logits shape:", tuple(logits.shape), "- raw logits; CrossEntropyLoss applies log-softmax itself")
"""),

    md("""
### Position embeddings: shuffle the patches
"""),
    code("""
p = 8
model = train.build_model(configs["p8"]).eval()
x = split["x_val"][:1]
g = p and 32 // p
perm = torch.randperm(g * g, generator=torch.Generator().manual_seed(0))
blocks = x.reshape(1, 3, g, p, g, p).permute(0, 1, 2, 4, 3, 5).reshape(1, 3, g * g, p, p)[:, :, perm]
shuffled = blocks.reshape(1, 3, g, g, p, p).permute(0, 1, 2, 4, 3, 5).reshape(1, 3, 32, 32)

with torch.no_grad():
    with_pos = (model(x) - model(shuffled)).abs().max().item()
    model.pos_embed.zero_()
    without_pos = (model(x) - model(shuffled)).abs().max().item()
print("(untrained model with its initial position embedding, std 0.02, so the effect is small")
print(" but clearly non-zero; training makes the embedding larger and more structured)")
print(f"largest logit change when the 16 patches are shuffled, with position embedding:    {with_pos:.4f}")
print(f"largest logit change when the 16 patches are shuffled, position embedding zeroed:  {without_pos:.2e}")
"""),

    md("""
## 4. Train both models from scratch

Adam, learning rate 3e-4, batch size 64, five epochs, the brief's starting
configuration. Validation runs with dropout off and no gradient updates.
"""),
    code("""
results = {name: train.run(cfg, split) for name, cfg in configs.items()}
"""),

    md("""
### Do the saved results reproduce?

`results/p4.json` and `results/p8.json` are what the report is built from.
Here, the live run is compared with them field by field.
"""),
    code("""
for name, live in results.items():
    saved = json.loads(Path(f"results/{name}.json").read_text())
    diffs = {k: max(abs(a[k] - b[k]) for a, b in zip(live["history"], saved["history"]))
             for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    same_preds = live["val_preds"] == saved["val_preds"]
    print(f"{name}: max |live - saved| per metric {diffs}; predictions identical: {same_preds}")
    print(f"      runtime live {live['total_seconds']:.1f} s vs saved {saved['total_seconds']:.1f} s "
          f"(wall-clock varies run to run; metrics do not)")
"""),

    md("### Learning curves, one model at a time"),
    code("""
for name, run in results.items():
    plots.learning_curves(run, name, out / f"{name}_curves.png")
    display(Image(filename=out / f"{name}_curves.png"))
"""),
    md("### Eight fixed validation examples, both models"),
    code("""
for name, run in results.items():
    plots.predictions_panel(run, split["raw_val"], out / f"{name}_predictions.png")
    display(Image(filename=out / f"{name}_predictions.png"))
print(pd.DataFrame({
    "true": [e["true"] for e in results["p4"]["fixed_examples"]],
    "4x4 predicted": [e["pred"] for e in results["p4"]["fixed_examples"]],
    "8x8 predicted": [e["pred"] for e in results["p8"]["fixed_examples"]],
}).to_string())
"""),

    md("""
## 5. The controlled comparison

Same split, seed, preprocessing, width, depth, heads, feed-forward width,
optimizer settings, batch size and epoch budget. Only the patch size differs.
"""),
    code("""
table = pd.DataFrame([{
    "patch": f"{r['config']['patch_size']}x{r['config']['patch_size']}",
    "N": r["model"]["num_patches"], "L": r["model"]["tokens"],
    "L^2 entries/head/layer": r["model"]["attn_entries"],
    "parameters": r["model"]["parameters"],
    "val acc": round(r["final_val_acc"], 4), "val loss": round(r["final_val_loss"], 4),
    "train acc (eval mode)": round(r["history"][-1]["train_eval_acc"], 4),
    "runtime (s)": round(r["total_seconds"], 1),
} for r in results.values()])
display(table)
ratio = table["L^2 entries/head/layer"][0] / table["L^2 entries/head/layer"][1]
print(f"attention-score entries, 4x4 / 8x8 = {ratio:.1f}x; runtime ratio = "
      f"{results['p4']['total_seconds'] / results['p8']['total_seconds']:.2f}x")
plots.comparison_curves(results, out / "comparison.png")
display(Image(filename=out / "comparison.png"))
"""),

    md("""
## 6. Supplementary evidence (saved by `scripts/run_supplementary.py` and `scripts/run_timing.py`)

Read from files rather than recomputed here: five seeds on the same split,
and per-step timing on CPU and the GPU.
"""),
    code("""
seeds = json.loads(Path("results/supplementary/seed_replication.json").read_text())["by_run"]
df = pd.DataFrame({"seed": seeds["p4"]["seeds"], "4x4 val acc": seeds["p4"]["val_acc"],
                   "8x8 val acc": seeds["p8"]["val_acc"]})
df["4x4 - 8x8 (points)"] = (100 * (df["4x4 val acc"] - df["8x8 val acc"])).round(2)
display(df)
print("4x4 ahead in", int((df["4x4 - 8x8 (points)"] > 0).sum()), "of", len(df), "seeds")
timing = json.loads(Path("results/supplementary/timing.json").read_text())
display(pd.DataFrame(timing["rows"]).round(3))
display(pd.DataFrame(timing["memory"]).round(2))
print("rows are one optimizer step (ms); memory is MB held after one forward pass on the GPU")
"""),

    md("""
## 7. Reading the result

The 4x4 model was ahead at every seed, by a modest margin, while computing
about 14.6 times as many attention-score entries per head and layer - and
without being measurably slower or faster on the GPU. Absolute accuracy is
low for both: the models are under-trained, not over-fitted. The full
400-600 word analysis, with the Swin discussion, the limits of this
experiment and the proposed follow-up, is in
`report/ViT_Patch_Size_Report.pdf`.

Reproduce everything with:

```
.venv/bin/python scripts/run_all.py
```
"""),
]


def build() -> Path:
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nbf.write(notebook, str(NOTEBOOK))
    return NOTEBOOK


def execute() -> int:
    """Execute in place, so the committed outputs are the ones this produced."""
    print(f"Executing {NOTEBOOK.name} (this retrains both models)")
    result = subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=1200", str(NOTEBOOK)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        return result.returncode
    print(f"Executed {NOTEBOOK.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = build()
    print(f"Wrote {path.relative_to(REPO_ROOT)} ({len(CELLS)} cells)")
    return execute() if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
