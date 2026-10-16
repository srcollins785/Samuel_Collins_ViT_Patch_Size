# Samuel_Collins_ViT_Patch_Size

A small Vision Transformer trained from scratch on a reproducible subset of
CIFAR-10, with one controlled comparison: 4x4 versus 8x8 patches, every other
setting held fixed.

Lab Assignment 6, Computer Vision. Clark Atlanta University, Department of
Cyber-Physical Systems. Instructor: Dr. Kishor Datta Gupta.

**Status: planning complete, implementation not started.** See
[PLAN.md](PLAN.md) for the phased plan, decisions, expected parameter counts,
and the run-file schema the report is generated from.

The report will be generated from `results/*.json` by
`scripts/generate_report.py` through the query functions in
`scripts/report_data.py` (already implemented and tested), so no number in it
is typed by hand.

## Installation (target)

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

## Running everything (target)

    .venv/bin/python scripts/run_all.py
    .venv/bin/python scripts/run_all.py --skip-training   # rebuild documents only

## Tests

    .venv/bin/python -m pytest

## License

MIT. See `LICENSE`.
