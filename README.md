# Samuel_Collins_ViT_Patch_Size

A small Vision Transformer trained from scratch on a reproducible subset of
CIFAR-10, with one controlled comparison: 4x4 versus 8x8 patches, every other
setting held fixed.

Lab Assignment 6, Computer Vision. Clark Atlanta University, Department of
Cyber-Physical Systems. Instructor: Dr. Kishor Datta Gupta.

**Result.** The 4x4 model reached 38.1% validation accuracy against 34.8% for
8x8, and was ahead in all five seeds tried (mean margin 2.5 points). It
computes 14.6 times as many attention-score entries per head and layer, yet its
measured training time was no longer: at this scale the cost is not set by the
attention map. Both models are under-trained at five epochs (about 40% accuracy
on their own training data), as expected of a tiny ViT.

**Supplementary: trained to convergence.** With early stopping on validation
(five seeds, same split and settings), the 4x4 lead widens to about 10 points
(57.1% against 47.0%), and a parameter-matched CNN reaches 63.2%, ahead of both.
This is validation-only and the test split is never loaded; see Section 4.6 of the
report for the caveats.

- Report: `report/ViT_Patch_Size_Report.pdf`
- Analysis alone (400-600 words): `report/ViT_Patch_Size_Analysis.pdf`
- Executed notebook: `ViT_Patch_Size.ipynb`
- Machine-written configuration: `configuration.yaml`

## What this repository is arranged to prove

- **The comparison is controlled.** `tests/test_train.py` and
  `scripts/validate_results.py` diff the two run configurations and fail unless
  `patch_size` is the only field that differs, and unless both recorded the same
  split checksum.
- **The model is what the report says it is.** `tests/test_model.py` asserts the
  tensor shapes, the parameter counts (75,082 and 81,226, a hand calculation
  written down before the code), that the hand-written attention equals
  `nn.MultiheadAttention`, that there is no causal mask, and that position
  embeddings are what make the output order-dependent. Four deliberate breakages
  (a causal mask, no position embedding, no sqrt(d) scaling, the wrong MLP width)
  each fail the suite; a test that cannot fail establishes nothing.
- **The report cannot state a result that was not produced.** Every number in
  `report/` is a query into `scripts/report_data.py` over `results/*.json`. The
  prose carries qualitative premises (for example "4x4 led in every seed");
  `generate_report.py` asserts them and refuses to build if a rerun changes one.
  The audit also enforces the 400-600 word analysis.

## Installation

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

No GPU is required. The device is selected automatically (MPS, CUDA or CPU) and
recorded in each run file. Chrome is needed only for the PDF.

## Running everything

    .venv/bin/python scripts/run_all.py

Tests, split, both graded runs, the five-seed replication, timing, the extended
training runs, audit, configuration, report, both PDFs and the executed notebook.
About ten minutes on an Apple M4 Max. The first failure stops the run.

    .venv/bin/python scripts/run_all.py --skip-training

Rebuilds the documents from existing results.

### Individual steps

    .venv/bin/python scripts/build_cifar_split.py   # download, record the split
    .venv/bin/python run_experiment.py --run both   # the two graded runs
    .venv/bin/python scripts/run_supplementary.py   # five-seed replication
    .venv/bin/python scripts/run_timing.py          # per-step timing and memory, CPU and GPU
    .venv/bin/python scripts/run_extended.py        # ViT 4x4, 8x8 and a CNN to convergence
    .venv/bin/python scripts/validate_results.py    # audit (add --report for the report)
    .venv/bin/python scripts/generate_report.py     # Markdown, from results/
    .venv/bin/python scripts/build_report_pdf.py    # PDF, via headless Chrome
    .venv/bin/python scripts/build_notebook.py --execute

## The data

The images are not in the repository; the brief forbids submitting dataset
downloads. Committed is `data/split_indices.json`: the 10,000 training and 2,000
validation indices (stratified, disjoint, from the official training split), the
seed, per-class counts, normalization statistics and a checksum.
`scripts/build_cifar_split.py` downloads CIFAR-10 and rebuilds the same split.
The official test split is never read by any code here.

## Documented choices

- **Left open by the brief:** dropout 0.1, pre-norm blocks, GELU, learned position
  embeddings, truncated-normal(0.02) initialization, per-channel normalization
  from the training subset only, no augmentation. All recorded in
  `configuration.yaml`.
- **Adam, lr 3e-4, batch 64, 5 epochs**: the brief's starting configuration, unchanged.
- **Compute option not used:** the full 10,000 / 2,000 / 5-epoch protocol runs in seconds.
- **Extended runs choose the stopping epoch on validation**, so their best-epoch
  accuracies are slightly optimistic, equally for every arm; accuracy at fixed epoch
  budgets involves no selection and is reported alongside.
- **Fixed display examples** are chosen from the seed and labels before any model
  is trained, so they are not selected to show errors.
- **Runtime is noisy** (about a second run to run); accuracies and predictions
  reproduce bit for bit.

## Layout

    src/vit_patch_size/   config, data, model, cnn, train, extended, evaluate, plots, env
    scripts/              split, supplementary, timing, audit, report, PDF, notebook, packaging
    tests/                pytest suite (64 tests)
    run_experiment.py     entry point for the two graded runs
    ViT_Patch_Size.ipynb  executed notebook, a thin wrapper over src/
    data/                 split_indices.json (committed); the CIFAR download is ignored
    results/ plots/ report/   generated; force-added once the final runs complete

## Run-file schema

`results/p4.json` and `results/p8.json` are the contract between the experiments
and the report. `report_data.load` rejects a file that violates it.

    config:  patch_size, embed_dim, depth, heads, mlp_dim, dropout, lr, batch_size,
             epochs, seed, n_train, n_val, split_checksum, optimizer, normalization
    model:   num_patches, tokens, attn_entries, parameters, parameter_breakdown, shapes{...}
    history: [{epoch, train_loss, train_acc, val_loss, val_acc, seconds, train_eval_acc}] x epochs
    total_seconds, final_val_acc, fixed_examples[{index, true, pred, prob}], val_preds,
    hardware{}, software{}

## Tests

    .venv/bin/python -m pytest

## Sources

No code was copied from another repository; the model is written from
Dosovitskiy et al. (ICLR 2021) and Vaswani et al. (NeurIPS 2017).
`nn.MultiheadAttention` is used only as a test oracle. The full source list,
including the two course lectures, is in Section 7 of the report.

## License

MIT. See `LICENSE`.
