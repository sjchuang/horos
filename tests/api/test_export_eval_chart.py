"""E6-T14: the evaluation sheet — one image with the confusion matrix beside
the per-class performance table.

Both halves come from the same error analysis the evaluate page draws, at the
threshold and IoU asked for, so the sheet and the page cannot disagree."""

from __future__ import annotations

import pytest
from helpers.runs import completed_fake_run

from horos.api.error_analysis import analyze_errors
from horos.api.evaluate import evaluate_run
from horos.api.export import EVAL_CHART_FORMATS, export_evaluation_chart
from horos.errors import ProjectError

pytest.importorskip("matplotlib", reason="report rendering needs matplotlib")
pytest.importorskip("pycocotools", reason="training stack not installed")


@pytest.fixture
def evaluated(tmp_path):
    project, record = completed_fake_run(tmp_path, epochs=1)
    evaluate_run(project, record.run_id, split="valid")
    return project, record


def test_a_sheet_is_written_for_each_format(evaluated, tmp_path):
    project, record = evaluated
    assert EVAL_CHART_FORMATS == ("png", "pdf")
    for fmt in EVAL_CHART_FORMATS:
        out = export_evaluation_chart(
            project, record.run_id, "valid", format=fmt,
            out_path=tmp_path / f"sheet.{fmt}",
        )
        assert out.is_file() and out.stat().st_size > 5_000
    header = (tmp_path / "sheet.png").read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "sheet.pdf").read_bytes()[:5] == b"%PDF-"


def test_it_lands_in_the_run_exports_by_default(evaluated):
    project, record = evaluated
    out = export_evaluation_chart(project, record.run_id, "valid")
    assert out.parent == project.root / "runs" / record.run_id / "exports"
    assert out.name == "evaluation_valid.png"


def test_the_sheet_follows_the_threshold_it_was_asked_for(evaluated, tmp_path):
    project, record = evaluated
    # the two thresholds give different analyses, so different sheets
    low = export_evaluation_chart(project, record.run_id, "valid", threshold=0.1,
                                  out_path=tmp_path / "low.png")
    high = export_evaluation_chart(project, record.run_id, "valid", threshold=0.95,
                                   out_path=tmp_path / "high.png")
    assert low.read_bytes() != high.read_bytes()
    # and the numbers on them are the analysis the page would show
    assert analyze_errors(project, record.run_id, "valid", threshold=0.1).fp \
        != analyze_errors(project, record.run_id, "valid", threshold=0.95).fp


def test_an_unknown_format_is_refused(evaluated):
    project, record = evaluated
    with pytest.raises(ProjectError, match="Unsupported evaluation chart format"):
        export_evaluation_chart(project, record.run_id, "valid", format="xlsx")


def test_a_split_with_no_evaluation_says_so(tmp_path):
    project, record = completed_fake_run(tmp_path, epochs=1)
    with pytest.raises(ProjectError, match="run an evaluation first"):
        export_evaluation_chart(project, record.run_id, "valid")


def test_many_classes_still_render(evaluated, tmp_path):
    """The table keeps the biggest classes and the matrix shrinks its type;
    neither may raise on a project with more classes than rows."""
    from horos.core.dataset import Category

    project, record = evaluated
    project.set_categories(
        list(project.categories)
        + [Category(id=10 + n, name=f"extra_{n}") for n in range(40)]
    )
    out = export_evaluation_chart(project, record.run_id, "valid",
                                  out_path=tmp_path / "many.png")
    assert out.is_file() and out.stat().st_size > 5_000
