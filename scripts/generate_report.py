"""Build the assignment report in Markdown from the saved run files.

Reads ``results/`` and ``data/split_indices.json``; writes ``report/``. It
imports nothing from the package: the experiments talk to the report through
files, so the document can be rebuilt in a second without retraining and can
only describe results that were produced.

The analysis (section 5) is written prose; every number inside it is a call
into ``report_data``. Prose needs qualitative premises - "the 4x4 model was
ahead in every seed", "the 8x8 model was not faster on the GPU" - and those
are asserted in ``check_premises`` before anything is written. If a rerun
changes a premise the build stops and says which, instead of publishing a
sentence the data no longer support.

Usage
-----
    .venv/bin/python scripts/generate_report.py
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_data as rd  # noqa: E402

REPORT_DIR = REPO_ROOT / "report"
REPORT_MD = REPORT_DIR / "ViT_Patch_Size_Report.md"

STUDENT_NAME = "Samuel Collins"
STUDENT_TITLE = "Ph.D. Student, Department of Cyber-Physical Systems"
INSTITUTION = "Clark Atlanta University"
STUDENT_EMAIL = "samuel.collins@students.cau.edu"
COURSE = "Computer Vision - Lab Assignment 6: Vision Transformers (ViT)"
INSTRUCTOR = "Dr. Kishor Datta Gupta"
REPOSITORY = "https://github.com/srcollins785/Samuel_Collins_ViT_Patch_Size"

ANALYSIS_HEADING = "## 5. Analysis"


# ---------------------------------------------------------------------------
# Premises the prose depends on
# ---------------------------------------------------------------------------

def check_premises(data: dict) -> None:
    stats = rd.seed_stats(data)
    problems = []
    if stats is None:
        problems.append("no seed replication (run scripts/run_supplementary.py)")
    if data.get("timing") is None:
        problems.append("no timing measurements (run scripts/run_timing.py)")
    if problems:
        raise SystemExit("Cannot write the report: " + "; ".join(problems))

    if rd.higher_final_val_acc(data) != "p4":
        problems.append("the graded 4x4 run no longer has the higher validation accuracy")
    if stats["p4_wins"] != stats["n"]:
        problems.append("4x4 is no longer ahead in every replicated seed")
    if not rd.graded_pair_in_replication(data):
        problems.append("the seed-42 replication does not reproduce the graded runs")
    if not (0.8 <= rd.timing_ratio(data, "mps") <= 1.25):
        problems.append("MPS step times are no longer about equal across patch sizes")
    if rd.timing_ratio(data, "cpu") < 1.5:
        problems.append("CPU step time no longer scales visibly with token count")
    if max(rd.generalization_gap(data, r) for r in rd.RUNS) > 0.05:
        problems.append("the train/validation gap is large; 'underfitting' no longer fits")
    for r in rd.RUNS:
        wrong = sum(e["true"] != e["pred"] for e in data["runs"][r]["fixed_examples"])
        if not 0 < wrong < len(data["runs"][r]["fixed_examples"]):
            problems.append(f"{r}: fixed examples are all right or all wrong; revisit section 6")
    for r in rd.RUNS:
        h = data["runs"][r]["history"]
        if not h[-1]["val_loss"] < h[-2]["val_loss"]:
            problems.append(f"{r}: validation loss no longer falls in the last epoch")
    if rd.runtime_ratio(data) > 2.0:
        problems.append("graded runtime ratio is far from 1; revisit the runtime paragraph")
    if problems:
        raise SystemExit(
            "The analysis prose was written for a different result. Revisit "
            "section 5 before regenerating:\n  - " + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def table(headers: list, rows: list) -> list:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def shape(s: list) -> str:
    return "(" + ", ".join(["B"] + [str(v) for v in s[1:]]) + ")"


def pts(x: float, places: int = 1) -> str:
    """A difference of accuracies, in percentage points."""
    return f"{100 * x:.{places}f}"


def pp(patch: int) -> str:
    return f"{patch}x{patch}"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def headline(data: dict) -> str:
    s = rd.seed_stats(data)
    return (
        f"**Smaller patches won here, by a modest margin, and not by being cheaper.** "
        f"The 4x4 model reached {rd.pct(rd.final(data, 'p4', 'val_acc'))} validation "
        f"accuracy against {rd.pct(rd.final(data, 'p8', 'val_acc'))} for 8x8 "
        f"(a gap of {pts(rd.val_acc_gap(data))} points), and it was ahead in "
        f"{s['p4_wins']} of {s['n']} seeds, by {pts(s['diff_mean'])} points on average "
        f"(range {pts(s['diff_min'])} to {pts(s['diff_max'])}). The 4x4 model computes "
        f"{rd.attention_entry_ratio(data):.1f} times as many attention-score entries per "
        f"head and layer, yet its measured training time was "
        f"{rd.seconds(data['runs']['p4']['total_seconds'])} against "
        f"{rd.seconds(data['runs']['p8']['total_seconds'])}: at this scale the cost is "
        f"not set by the attention map. Absolute accuracy is low for both "
        f"(below {math.ceil(max(rd.final(data, r, 'val_acc') for r in rd.RUNS) * 20) * 5} percent), "
        f"as expected of a tiny ViT trained from scratch for five epochs."
    )


def header(data: dict) -> list:
    p4 = data["runs"]["p4"]
    manifest = data["manifest"]
    return [
        "# Patch Size in a Small Vision Transformer",
        "",
        f"**{STUDENT_NAME}** - {STUDENT_TITLE}, {INSTITUTION}",
        f"{STUDENT_EMAIL}",
        "",
        f"{COURSE} - Instructor: {INSTRUCTOR} - {date.today().isoformat()}",
        "",
        f"Repository: {REPOSITORY}",
        "",
        "---",
        "",
        "## Summary",
        "",
        headline(data),
        "",
        f"A two-block ViT (width 64, four heads, feed-forward width 128) was trained "
        f"from scratch on a stratified {manifest['n_train']:,}-image subset of "
        f"CIFAR-10 with {manifest['n_val']:,} separate validation images, for "
        f"{rd.config(data, 'p4', 'epochs')} epochs, once with 4x4 patches and once "
        f"with 8x8 patches. The split, seed, preprocessing, optimizer, batch size, "
        f"epoch budget and every width, depth and head count were identical; only the "
        f"patch size differed. The official test split was never loaded. Total "
        f"training time was {rd.seconds(sum(r['total_seconds'] for r in data['runs'].values()))} "
        f"on {p4['hardware']['chip']} ({p4['hardware']['device']}).",
        "",
        "Every table, figure and number in this report was generated from the saved "
        "run files by `scripts/generate_report.py`; none was typed by hand. "
        "`scripts/validate_results.py` audits those files, including that the two run "
        "configurations differ in the patch size and in no other field.",
        "",
    ]


def section_data(data: dict) -> list:
    m = data["manifest"]
    c = data["runs"]["p4"]["config"]
    hw, sw = data["runs"]["p4"]["hardware"], data["runs"]["p4"]["software"]
    rows = [
        ("Dataset", f"CIFAR-10 official training split ({m['population']:,} images), native 32 x 32 RGB"),
        ("Training subset", f"{m['n_train']:,} images, {m['train_per_class'][0]:,} per class"),
        ("Validation subset", f"{m['n_val']:,} images, {m['val_per_class'][0]} per class, disjoint from training"),
        ("Test split", "never loaded; kept out of model selection"),
        ("Seed", c["seed"]),
        ("Split checksum", f"`{m['checksum'][:16]}...` (SHA-256 over both index lists)"),
        ("Split indices", "`data/split_indices.json` (committed)"),
        ("Normalization", f"per-channel mean {[round(v, 4) for v in m['mean']]}, std "
                          f"{[round(v, 4) for v in m['std']]}, from the training subset only"),
        ("Augmentation", "none"),
        ("Optimizer", f"Adam, learning rate {c['lr']:g}, betas {tuple(c['betas'])}, no weight decay"),
        ("Batch size", f"{c['batch_size']} ({c['batches_per_epoch']} batches per epoch)"),
        ("Epochs", c["epochs"]),
        ("Dropout", f"{c['dropout']} in attention, MLP and embeddings; off at evaluation"),
        ("Loss", "`CrossEntropyLoss` on raw logits (log-softmax is inside the loss; no second softmax)"),
        ("Hardware", f"{hw['chip']}, device `{hw['device']}`, {hw['platform']}"),
        ("Libraries", f"Python {sw['python']}, torch {sw['torch']}, torchvision {sw['torchvision']}, numpy {sw['numpy']}"),
    ]
    return [
        "## 1. Data and configuration",
        "",
        "The images are not in the repository (the brief forbids submitting dataset "
        "downloads). `scripts/build_cifar_split.py` rebuilds the same split from the "
        "committed indices, and both runs recorded the same checksum, which the audit "
        "checks. All settings below are identical for the two runs except the patch size.",
        "",
        *table(["Item", "Value"], rows),
        rd.figure("plots/samples.png", "Sixteen labeled training examples."),
        "",
    ]


def section_model(data: dict) -> list:
    a, b = data["runs"]["p4"], data["runs"]["p8"]
    sa, sb = a["model"]["shapes"], b["model"]["shapes"]
    stages = [("images", "Images"), ("patches", "Flattened non-overlapping patches"),
              ("projected_tokens", "Projected tokens (shared linear layer)"),
              ("with_class_token", "With class token"),
              ("with_position_embedding", "With position embedding"),
              ("encoder_output", "Encoder output (2 blocks)"),
              ("class_representation", "Class-token representation"),
              ("logits", "Logits")]
    shape_rows = [(label, shape(sa[k]), shape(sb[k])) for k, label in stages]

    values = [[0.0, 0.0, 0.0], [1.0, 4.0, 0.0], [1.0, 3.0, 1.5]]
    scores = [2.0, 4.0, 4.0]

    def softmax(xs):
        e = [math.exp(x) for x in xs]
        return [v / sum(e) for v in e]

    def mix(w):
        return [sum(w[i] * values[i][j] for i in range(3)) for j in range(3)]

    w_raw = softmax(scores)
    w_scaled = softmax([s / math.sqrt(3) for s in scores])
    f3 = lambda xs: "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"

    return [
        "## 2. The model",
        "",
        "The model follows Dosovitskiy et al. (2021) at the sizes the brief fixes: "
        "non-overlapping patches, one shared linear projection to width 64, a learnable "
        "class token, learned position embeddings, two pre-norm encoder blocks "
        "(LayerNorm, four-head self-attention, residual; LayerNorm, MLP of width 128 "
        "with GELU, residual), a final LayerNorm and a linear head to 10 logits. The "
        "encoder is written out by hand (`src/vit_patch_size/model.py`) so the "
        "attention weights are available; `tests/test_model.py` checks it against "
        "`nn.MultiheadAttention` with identical weights, so it is verified, not trusted.",
        "",
        "### 2.1 Tensor shapes",
        "",
        "Shapes below were recorded from a real forward pass, not derived from "
        "constructor arguments. B is the batch size.",
        "",
        *table(["Stage", f"{pp(4)} patches", f"{pp(8)} patches"], shape_rows),
        f"With P x P patches on a 32 x 32 image, the number of patch tokens is "
        f"N = (32/P)^2 and the total including the class token is L = N + 1. "
        f"For P = 4, N = (32/4)^2 = {a['model']['num_patches']} and L = {a['model']['tokens']}. "
        f"For P = 8, N = (32/8)^2 = {b['model']['num_patches']} and L = {b['model']['tokens']}. "
        f"Each flattened patch has 3 x P x P values: {sa['patches'][2]} for 4x4 and "
        f"{sb['patches'][2]} for 8x8.",
        "",
        "### 2.2 Queries, keys and values",
        "",
        "Each token t is multiplied by three learned matrices to give a query "
        "q = W_Q t, a key k = W_K t and a value v = W_V t. The query and key decide "
        "*where to look*: the score of token i for token j is the dot product q_i . k_j, "
        "a similarity (large when the two vectors point the same way). Scores are "
        "divided by the square root of the key dimension, for stable gradients, and "
        "passed through a softmax along j, giving non-negative weights that sum to 1. "
        "The value decides *what is passed on*: the output for token i is the "
        "weight-weighted sum of every token's value vector. Four heads repeat this in "
        "four 16-dimensional subspaces and an output projection mixes the results "
        "(lecture L2, slides 21-35).",
        "",
        "**A worked example with three value vectors** (the values and the scores "
        "[2, 4, 4] for query 1 are the lecture L1 example):",
        "",
        *table(["", "weights", "weighted sum of v1=[0,0,0], v2=[1,4,0], v3=[1,3,1.5]"],
               [("softmax of raw scores", f3(w_raw), f3(mix(w_raw))),
                ("softmax of scores / sqrt(3) (what the model does)", f3(w_scaled), f3(mix(w_scaled)))]),
        "The lecture slide rounds the softmax to [0.0, 0.5, 0.5], which gives [2.0, 7.0, 1.5]; "
        "the exact values are above. Token 1 gets almost no weight because its key "
        "matches the query worst, so the output is nearly an even blend of v2 and v3. "
        "The slide's last summary row reads [2.0, 7.0, 1.0] where its own step gives "
        "1.5, so this report does not reuse that row. `tests/test_model.py` asserts "
        "these exact numbers.",
        "",
        "### 2.3 Why positions are needed, and why there is no causal mask",
        "",
        "Self-attention is permutation-equivariant: the weights depend only on "
        "token content, so without extra information the patches are an unordered "
        "set and a shuffled image looks the same as the original. The learned "
        "position embedding, added to each token, is what tells the model where a "
        "patch came from. This is tested rather than assumed: with the embedding "
        "zeroed, shuffling an image's patches leaves the logits unchanged, and with "
        "a non-zero embedding it does not "
        "(`test_position_embedding_breaks_permutation_symmetry`).",
        "",
        "Image classification has no left-to-right generation order, so nothing "
        "should be hidden: every patch may attend to every patch, in both "
        "directions. A causal mask (future positions set to minus infinity before "
        "the softmax) belongs to the transformer *decoder* (lecture L2, decoder "
        "slide); the ViT uses only the encoder. With a causal mask the class token, "
        "at position 0, could attend to nothing but itself. `tests/test_model.py` "
        "checks that attention above the diagonal is non-zero and that changing "
        "only the last patch changes the class output.",
        "",
    ]


def history_rows(run: dict) -> list:
    return [(h["epoch"], rd.fmt(h["train_loss"]), rd.pct(h["train_acc"]),
             rd.pct(h["train_eval_acc"]), rd.fmt(h["val_loss"]), rd.pct(h["val_acc"]),
             f"{h['seconds']:.2f}") for h in run["history"]]


def examples_table(run: dict) -> list:
    return table(["#", "true", "predicted", "confidence", "correct"],
                 [(i + 1, e["true"], e["pred"], f"{e['prob']:.2f}",
                   "yes" if e["true"] == e["pred"] else "no")
                  for i, e in enumerate(run["fixed_examples"])])


def section_p4(data: dict) -> list:
    r = data["runs"]["p4"]
    wrong = sum(e["true"] != e["pred"] for e in r["fixed_examples"])
    return [
        "## 3. The 4x4 model: training and evidence",
        "",
        f"Trained from scratch for {r['config']['epochs']} epochs with the brief's starting "
        f"configuration (Adam, learning rate 3e-4, batch size 64), unchanged. "
        f"Validation ran with dropout disabled (`model.eval()`) and no gradient "
        f"updates (`torch.no_grad()`); a test checks that evaluation changes no "
        f"parameter.",
        "",
        f"**Final validation accuracy {rd.pct(r['final_val_acc'])}**, "
        f"validation loss {rd.fmt(r['final_val_loss'])}, "
        f"**{r['model']['parameters']:,} parameters**, "
        f"**training runtime {rd.seconds(r['total_seconds'])}** (training and validation, "
        f"{r['config']['epochs']} epochs, {r['hardware']['device']}). Chance is 10%.",
        "",
        *table(["epoch", "train loss*", "train acc*", "train acc (eval mode)",
                "val loss", "val acc", "seconds"], history_rows(r)),
        "\\* Running mean over the epoch's batches, measured with dropout active while "
        "the weights were changing. The 'eval mode' column scores the whole training "
        "subset after the epoch with dropout off, and is not included in the seconds.",
        "",
        rd.figure("plots/p4_curves.png", "4x4 model: training and validation loss and accuracy per epoch."),
        "",
        f"Eight fixed validation examples (one per class for eight classes, chosen "
        f"from the seed and the labels alone - never from a model's predictions - so "
        f"both models are shown the same images). {wrong} of the 8 are misclassified, "
        f"so mistakes are shown.",
        "",
        rd.figure("plots/p4_predictions.png", "4x4 model on the eight fixed validation examples."),
        "",
        *examples_table(r),
    ]


def section_comparison(data: dict) -> list:
    a, b = data["runs"]["p4"], data["runs"]["p8"]
    s = rd.seed_stats(data)
    agree = rd.agreement(data)
    gap = rd.largest_class_gap(data)

    results = [
        (pp(r["config"]["patch_size"]), r["model"]["num_patches"], r["model"]["tokens"],
         f"{r['model']['parameters']:,}", rd.pct(r["final_val_acc"]),
         rd.fmt(r["final_val_loss"]), rd.pct(rd.final(data, k, "train_eval_acc")),
         rd.seconds(r["total_seconds"]))
        for k, r in (("p4", a), ("p8", b))
    ]
    breakdown = [(label, f"{a['model']['parameter_breakdown'][k]:,}",
                  f"{b['model']['parameter_breakdown'][k]:,}")
                 for k, label in (("patch_projection", "Patch projection (3 P^2 x 64 + 64)"),
                                  ("class_token", "Class token"),
                                  ("position_embedding", "Position embedding (L x 64)"),
                                  ("encoder_blocks", "Two encoder blocks"),
                                  ("final_norm_and_head", "Final LayerNorm and head"))]
    breakdown.append(("**Total**", f"**{a['model']['parameters']:,}**", f"**{b['model']['parameters']:,}**"))
    cost = [(pp(r["config"]["patch_size"]), r["model"]["num_patches"], r["model"]["tokens"],
             f"{r['model']['attn_entries']:,}", f"{r['model']['attn_entries_total']:,}")
            for r in (a, b)]
    timing_rows = []
    for dev in ("cpu", "mps"):
        r4, r8 = rd.timing_row(data, dev, 4), rd.timing_row(data, dev, 8)
        if r4 and r8:
            timing_rows.append((dev.upper(), f"{r4['train_step_ms']:.2f}", f"{r8['train_step_ms']:.2f}",
                                f"{rd.timing_ratio(data, dev):.2f}x",
                                f"{r4['score_and_softmax_ms']:.3f}", f"{r8['score_and_softmax_ms']:.3f}",
                                f"{rd.timing_ratio(data, dev, 'score_and_softmax_ms'):.2f}x"))
    seed_rows = [(sd, rd.pct(x4), rd.pct(x8), f"{pts(d)}")
                 for sd, x4, x8, d in zip(s["seeds"], s["p4"], s["p8"], s["diffs"])]
    seed_rows.append(("**mean (sd)**", f"{rd.pct(s['p4_mean'])} ({pts(s['p4_sd'])})",
                      f"{rd.pct(s['p8_mean'])} ({pts(s['p8_sd'])})",
                      f"{pts(s['diff_mean'])} ({pts(s['diff_sd'])})"))
    timing = data["timing"]

    return [
        "## 4. Controlled patch-size comparison",
        "",
        "The 8x8 model was trained from scratch with the same split, preprocessing, "
        "seed, embedding width, depth, heads, feed-forward width, optimizer settings, "
        "batch size and epoch budget. `scripts/validate_results.py` diffs the two "
        "saved configurations and fails unless `patch_size` is the only field that "
        "differs. Both models also see the same batches in the same order: the loader's "
        "shuffle is seeded on the CPU, independent of device.",
        "",
        *table(["patch", "N", "L", "parameters", "val acc", "val loss",
                "train acc (eval mode)", "runtime"], results),
        rd.figure("plots/comparison_curves.png",
                  "Both learning curves: training (dashed) and validation (solid) loss and accuracy."),
        "",
        rd.figure("plots/p8_predictions.png", "8x8 model on the same eight fixed validation examples."),
        "",
        *table(["#", "true", "4x4 predicted", "8x8 predicted"],
               [(i + 1, x["true"], x["pred"], y["pred"]) for i, (x, y) in
                enumerate(zip(a["fixed_examples"], b["fixed_examples"]))]),
        "### 4.1 What changed in the parameters",
        "",
        "Only two shapes depend on the patch size. The patch projection maps "
        f"3 x P x P inputs to 64, so it grows from {sa(a, 'patch_projection')} to "
        f"{sa(b, 'patch_projection')} parameters; the position embedding has one 64-vector "
        f"per token, so it shrinks from {sa(a, 'position_embedding')} to "
        f"{sa(b, 'position_embedding')}. The projection grows by more than the embedding "
        f"shrinks, so **the 8x8 model has more parameters ({b['model']['parameters']:,} "
        f"against {a['model']['parameters']:,}) despite four times fewer tokens**: "
        f"each patch carries four times the pixels.",
        "",
        *table(["Component", pp(4), pp(8)], breakdown),
        "### 4.2 Attention cost",
        "",
        "Each head in each layer scores every token against every token, an L x L "
        "map with L^2 entries. This is the count the brief asks for; the last column "
        "multiplies it by 4 heads and 2 layers.",
        "",
        *table(["patch", "N = (32/P)^2", "L = N + 1", "L^2 entries per head and layer",
                "entries, 4 heads x 2 layers"], cost),
        f"The 4x4 model computes {a['model']['attn_entries']:,} / {b['model']['attn_entries']:,} "
        f"= **{rd.attention_entry_ratio(data):.1f} times** as many score entries. Its "
        f"measured training time was {rd.runtime_ratio(data):.2f} times the 8x8 model's, "
        f"not {rd.attention_entry_ratio(data):.1f}.",
        "",
        "### 4.3 Why the count does not predict runtime or memory",
        "",
        "L^2 counts one thing: entries of the score matrix. Total time and memory "
        "also include work that scales with the number of tokens, not its square "
        "(the patch projection, both LayerNorms, the QKV and output projections, the "
        "MLP, the residual adds - all linear in L), work that does not depend on L at "
        "all (the batch transfer, the optimizer update of all parameters, kernel "
        "launches, Python overhead), and memory dominated by parameters, Adam's two "
        "moment buffers and per-token activations rather than by the 65 x 65 map. "
        "Two measurements (`scripts/run_timing.py`, median of "
        f"{timing['steps']} optimizer steps at batch size {timing['batch_size']} "
        "after warm-up) show the effect directly:",
        "",
        *table(["device", "step ms 4x4", "step ms 8x8", "step ratio",
                "scores+softmax ms 4x4", "scores+softmax ms 8x8", "scores ratio"], timing_rows),
        f"On the CPU the whole training step costs {rd.timing_ratio(data, 'cpu'):.2f} times "
        f"as much at 4x4, and the isolated score-and-softmax computation "
        f"{rd.timing_ratio(data, 'cpu', 'score_and_softmax_ms'):.2f} times - both far "
        f"below {rd.attention_entry_ratio(data):.1f}, because at width 64 and 16 "
        f"dimensions per head these matrices are tiny and most time goes to per-token "
        f"and fixed work. On the GPU (`mps`) the step ratio is "
        f"{rd.timing_ratio(data, 'mps'):.2f}: the two patch sizes cost the same, "
        f"consistent with a step dominated by fixed per-step overhead rather than "
        f"arithmetic. I did not profile the GPU kernels, so the overhead explanation "
        f"is an inference from these ratios, not a measurement of the cause. Peak "
        f"memory was not measured.",
        "",
        "### 4.4 Is the accuracy gap more than seed noise?",
        "",
        f"A single accuracy on {data['seeds']['n_val']:,} validation images has a binomial "
        f"standard error of about {pts(s['binomial_se'])} points, so the graded "
        f"{pts(rd.val_acc_gap(data))}-point gap is only about "
        f"{rd.val_acc_gap(data) / s['binomial_se']:.1f} standard errors on its own. "
        f"I therefore repeated both patch sizes at {s['n']} seeds on the same split "
        f"(`scripts/run_supplementary.py`; the seed changes the initial weights, batch "
        f"order and dropout masks). The seed-42 pair reproduces the graded runs exactly. "
        f"This is supplementary; the graded comparison is the seed-42 pair.",
        "",
        *table(["seed", "4x4 val acc", "8x8 val acc", "difference (points)"], seed_rows),
        rd.figure("plots/seed_replication.png", "Final validation accuracy by seed, both patch sizes."),
        "",
        f"The 4x4 model is ahead in {s['p4_wins']} of {s['n']} seeds, by "
        f"{pts(s['diff_mean'])} points on average, but the smallest margin is only "
        f"{pts(s['diff_min'])} points, and {s['n']} seeds cannot rule out an occasional "
        f"reversal.",
        "",
        "### 4.5 Where the models differ",
        "",
        f"The models are right together and wrong together more often than not: both "
        f"correct on {agree['both_right']:,} of {agree['n']:,} validation images, both wrong on "
        f"{agree['both_wrong']:,}, only 4x4 correct on {agree['only_p4_right']}, only 8x8 correct "
        f"on {agree['only_p8_right']}. The largest per-class difference is **{gap['class']}**: "
        f"{rd.pct(gap['p4'], 0)} at 4x4 against {rd.pct(gap['p8'], 0)} at 8x8, "
        f"{gap['images']} of the {gap['per_class']} {gap['class']} images.",
        "",
        rd.figure("plots/per_class_accuracy.png", "Per-class validation accuracy for both models (seed 42)."),
        "",
    ]


def sa(run: dict, key: str) -> str:
    return f"{run['model']['parameter_breakdown'][key]:,}"


def section_analysis(data: dict) -> list:
    a, b = data["runs"]["p4"], data["runs"]["p8"]
    s = rd.seed_stats(data)
    gap = rd.largest_class_gap(data)
    swin = rd.swin_window_entries()
    ratio = rd.attention_entry_ratio(data)

    agree = rd.agreement(data)
    paragraphs = [
        f"**Performance.** The 4x4 model finished at {rd.pct(rd.final(data, 'p4', 'val_acc'))} "
        f"validation accuracy and the 8x8 model at {rd.pct(rd.final(data, 'p8', 'val_acc'))}, "
        f"with validation loss {rd.fmt(rd.final(data, 'p4', 'val_loss'), 3)} against "
        f"{rd.fmt(rd.final(data, 'p8', 'val_loss'), 3)}. One pair is thin evidence, since one "
        f"accuracy carries about {pts(s['binomial_se'])} points of sampling error, but across "
        f"{s['n']} seeds 4x4 led every time, by {pts(s['diff_mean'])} points on average. "
        f"The advantage concentrated in particular classes: {gap['class']} accuracy was "
        f"{rd.pct(gap['p4'], 0)} against {rd.pct(gap['p8'], 0)}. Most examples behave alike: "
        f"both models were right on {agree['both_right']:,} images and wrong on "
        f"{agree['both_wrong']:,} of {agree['n']:,}, and they split on the remainder "
        f"({agree['only_p4_right']} only 4x4, {agree['only_p8_right']} only 8x8). Smaller "
        f"patches did not buy accuracy with time: runtimes were "
        f"{rd.seconds(a['total_seconds'])} and {rd.seconds(b['total_seconds'])}, within the "
        f"run-to-run noise of about a second, although 4x4 scores {ratio:.1f} times as "
        f"many attention entries. The cost of the larger map is real but invisible at "
        f"this scale.",

        "**Global attention and position.** A convolution sees a small neighborhood, "
        "and distant pixels meet only after many layers; lecture L2 counts O(n/k) layers "
        "for kernel size k, against O(1) for self-attention. Here every one of the 65 or 17 "
        "tokens attends to all others in the first block. A convolution also shares weights "
        "across positions, which builds in translation equivariance; a ViT has no such prior. "
        "Attention alone ignores order, so the position embedding is what lets the model use "
        "where a patch is; a test confirmed that without it, shuffling patches changes nothing.",

        f"**Poor absolute performance.** Both models are far from converged: "
        f"training accuracy scored in evaluation mode was {rd.pct(rd.final(data, 'p4', 'train_eval_acc'))} "
        f"for 4x4, barely above its validation accuracy, and the curves were still "
        f"falling at epoch five. That is underfitting, not overfitting. A ViT lacks a "
        f"CNN's built-in locality, so it must learn such structure from data, and the "
        f"lecture attributes ViT's success to pretraining on hundreds of millions of "
        f"images; this model saw ten thousand, briefly. The experiment ran no CNN, one "
        f"dataset, one tiny budget, so it says nothing about CNNs or transformers "
        f"being better in general.",

        f"**Swin for dense tasks.** Three separate ideas help. Windowed attention cuts "
        f"cost: on a {swin['tokens']:,}-token map (56 x 56, a 224-pixel image with 4x4 "
        f"patches), global attention needs {swin['global']:,} entries per head, while "
        f"7 x 7 windows need {swin['windowed']:,}, {swin['ratio']:.0f} times fewer; that "
        f"is arithmetic, not a measurement. Shifted windows restore communication, "
        f"because tokens in different windows interact in alternate layers. Patch "
        f"merging builds multiscale features, giving the detector or segmenter the "
        f"coarse-to-fine maps a flat coarse-patch ViT lacks; lecture L2 has the four stages "
        f"shrink a 224-pixel input to 56, 28, 14 and 7 tokens per side, the resolutions "
        f"of ResNet-style backbones. Small objects need fine patches, and windows make "
        f"fine patches affordable.",

        "**Next experiment.** Train both patch sizes for many more epochs, with "
        "early stopping on validation, alongside a parameter-matched small CNN under the "
        "same protocol and several seeds. That would show whether the patch-size gap "
        "persists at convergence and give the missing CNN reference. Comparing the "
        "patch sizes at several equal optimizer-step budgets would also separate slower "
        "convergence from a genuinely lower ceiling at 8x8.",
    ]
    return [ANALYSIS_HEADING, "", *[p + "\n" for p in paragraphs]]


def section_notes(data: dict) -> list:
    s = rd.seed_stats(data)
    lo4, hi4 = rd.seconds_range(data, "p4")
    lo8, hi8 = rd.seconds_range(data, "p8")
    return [
        "## 6. Limitations and honest notes",
        "",
        "- **Weak absolute accuracy.** Both models sit well under 50%, as the brief "
        "anticipates. Nothing here was tuned to raise them.",
        f"- **Runtime is noisy.** Across the {s['n']} replicated seeds the 4x4 run took "
        f"{lo4:.1f}-{hi4:.1f} s and the 8x8 run {lo8:.1f}-{hi8:.1f} s. Differences "
        f"under about a second are not resolvable, so no claim here depends on "
        f"which patch size was faster, only on neither being many times faster. "
        f"Accuracies and predictions reproduce bit for bit on rerun; wall-clock time does not.",
        "- **One split, two sizes, small validation set.** The split is fixed, so seed "
        "variation excludes split-to-split variation. The seed replication addresses "
        "initialization and batch order only.",
        "- **Training metrics.** The 'train' columns are running means with dropout on; "
        "the 'eval mode' column is the like-for-like comparison with validation.",
        "- **Fixed examples** are chosen before any model is trained, from the seed and "
        "labels, so they are not selected to show errors. They happen to include both "
        "correct and incorrect predictions.",
        "- **Compute option not used.** The full 10,000 / 2,000 / 5-epoch protocol ran "
        "in seconds, so the reduced option was unnecessary.",
        "- **Choices the brief left open**: dropout 0.1, pre-norm blocks, GELU, learned "
        "position embeddings, trunc-normal(0.02) initialization, per-channel "
        "normalization from the training subset, no augmentation. All recorded in "
        "`configuration.yaml` and identical across the two runs.",
        "- **Lecture slides.** Two slips in lecture L1's attention example are noted in "
        "section 2.2; this report uses exact values.",
        "",
    ]


def section_reproduce(data: dict) -> list:
    return [
        "## 7. Reproduction and sources",
        "",
        "```",
        "python3.12 -m venv .venv",
        ".venv/bin/python -m pip install -r requirements.txt",
        ".venv/bin/python scripts/run_all.py",
        "```",
        "",
        "`run_all.py` runs the tests, rebuilds the split, trains both models, runs the "
        "seed replication and timing, audits the results, writes `configuration.yaml`, "
        "builds this report and the executed notebook, and stops at the first failure. "
        "`--skip-training` rebuilds the documents only.",
        "",
        "**Sources.** No code was copied from another repository; the model is written "
        "from the papers below.",
        "",
        "- Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image "
        "Recognition at Scale.* ICLR 2021 (architecture, class token, learned position embeddings).",
        "- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017 (scaled dot-product "
        "attention, multi-head attention, residual and LayerNorm structure).",
        "- Liu et al. *Swin Transformer: Hierarchical Vision Transformer Using Shifted "
        "Windows.* ICCV 2021 (windows, shifting, patch merging).",
        "- Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* 2009 (CIFAR-10).",
        "- Kingma and Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.",
        "- PyTorch `nn.MultiheadAttention` documentation: used only as a test oracle "
        "for the hand-written attention, not in the model.",
        "- Course lectures: **L1** `VisionTransformers_Details.pdf` (ViT and Swin; "
        "cites Dosovitskiy et al. and Liu et al.) and **L2** `visiontransformers.pdf` "
        "(UCSD CSE 252D, SP24, Chandraker et al.; cites Alammar's *The Illustrated "
        "Transformer*).",
        "",
    ]


def build(data: dict) -> str:
    check_premises(data)
    parts = [
        header(data), section_data(data), section_model(data), section_p4(data),
        section_comparison(data), section_analysis(data), section_notes(data),
        section_reproduce(data),
    ]
    return "\n".join(line for part in parts for line in part).rstrip() + "\n"


def main() -> int:
    data = rd.load()
    manifest = REPO_ROOT / "data" / "split_indices.json"
    data["manifest"] = json.loads(manifest.read_text())
    REPORT_DIR.mkdir(exist_ok=True)
    REPORT_MD.write_text(build(data))
    words = analysis_words(REPORT_MD.read_text())
    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}; analysis is {words} words")
    return 0


def analysis_words(markdown: str) -> int:
    """Words in section 5 only, markup stripped. The audit enforces 400-600."""
    start = markdown.index(ANALYSIS_HEADING) + len(ANALYSIS_HEADING)
    end = markdown.find("\n## ", start)
    body = markdown[start:end if end != -1 else None]
    return len(body.replace("**", "").split())


if __name__ == "__main__":
    raise SystemExit(main())
