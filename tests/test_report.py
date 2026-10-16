"""The report refuses to publish prose the data no longer support."""

import copy

import pytest

import generate_report as gr
import report_data as rd

pytestmark = pytest.mark.skipif(
    not (rd.RESULTS_DIR / "p4.json").is_file() or not (rd.RESULTS_DIR / "p8.json").is_file(),
    reason="needs the committed results/",
)


@pytest.fixture
def data():
    d = rd.load()
    d["manifest"] = {}
    return d


def test_premises_hold_for_the_saved_results(data):
    gr.check_premises(data)


def test_premises_fail_when_the_ranking_flips(data):
    flipped = copy.deepcopy(data)
    flipped["runs"]["p4"]["history"][-1]["val_acc"] = 0.10
    with pytest.raises(SystemExit, match="higher validation accuracy"):
        gr.check_premises(flipped)


def test_premises_fail_without_replication(data):
    d = copy.deepcopy(data)
    d["seeds"] = None
    with pytest.raises(SystemExit, match="seed replication"):
        gr.check_premises(d)


def test_premises_fail_if_a_seed_reverses(data):
    d = copy.deepcopy(data)
    d["seeds"]["by_run"]["p4"]["val_acc"][3] = 0.10
    with pytest.raises(SystemExit, match="every replicated seed"):
        gr.check_premises(d)


def test_word_counter_counts_section_five_only():
    md = "## 4. x\nlots of words here\n## 5. Analysis\n\n**a** b c\n\n## 6. y\nmore more more more\n"
    assert gr.analysis_words(md) == 3


def test_standalone_analysis_is_the_same_text_as_section_five(data):
    import json
    data["manifest"] = json.loads((rd.REPO_ROOT / "data" / "split_indices.json").read_text())
    full = gr.build(data)
    alone = gr.build_analysis_document(data)
    assert "## Analysis" in alone and gr.ANALYSIS_HEADING not in alone
    section = full[full.index(gr.ANALYSIS_HEADING) + len(gr.ANALYSIS_HEADING):]
    section = section[:section.index("\n## ")].strip()
    assert section in alone
    assert 400 <= len(section.replace("**", "").split()) <= 600
