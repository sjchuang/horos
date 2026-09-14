"""The WebUI pages render and link to each other (R2: the UI is served by the
same app but only ever talks to /api/v1 from the browser)."""

from __future__ import annotations

import re

import pytest

from horos.web.app import create_app

PAGES = {
    "/": "index",
    "/annotate": "annotate",
    "/train": "train",
    "/evaluate": "evaluate",
    "/experiments": "experiments",
    "/lab": "lab",
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.mark.parametrize("path", sorted(PAGES))
def test_page_renders_and_links_to_every_other_page(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    html = response.get_data(as_text=True)
    for other in PAGES:
        assert f'href="{other}"' in html, f"{path} has no nav link to {other}"
    # exactly one nav entry is marked as the current page, and it is this one
    current = re.findall(r'<a href="([^"]+)"[^>]*aria-current="page"', html)
    assert current == [path]


def test_evaluate_page_holds_metrics_and_error_analysis_only(client):
    html = client.get("/evaluate").get_data(as_text=True)
    assert 'id="eval-btn"' in html and 'id="errors-panel"' in html
    assert 'id="dropzone"' not in html and 'id="gallery"' not in html
    assert 'href="/lab"' in html  # points the user to where uploads went


def test_lab_page_holds_the_upload_playground_only(client):
    html = client.get("/lab").get_data(as_text=True)
    assert 'id="dropzone"' in html and 'id="gallery"' in html
    assert "inst.segmentation" in html  # segmentation models show masks, not only boxes
    assert 'id="eval-btn"' not in html and 'id="errors-panel"' not in html


def test_pages_only_call_the_web_api(client):
    # R2: every fetch() in the inline scripts goes through the /api/v1 prefix
    for path in PAGES:
        html = client.get(path).get_data(as_text=True)
        for call in re.findall(r'fetch\(\s*"([^"]+)"', html):
            assert call.startswith("/api/v1"), f"{path} fetches {call} directly"


def test_experiments_page_holds_the_comparison_table_and_editor(client):
    # E7-T6: the run table, the side-by-side panel and the notes/tags editor
    html = client.get("/experiments").get_data(as_text=True)
    for element in ("runs-table", "sort-select", "ref-select", "compare-panel",
                    "notes-input", "tags-input", "save-btn", "link-train"):
        assert f'id="{element}"' in html, element
    # E7-S2: the export flow is entered from the table via the Training page's deep link
    assert "/train#" in html
    # every endpoint the page talks to exists under /api/v1/experiments
    assert "/experiments/runs" in html and "/experiments/compare" in html


def test_loop_page_is_the_four_step_shell_without_a_canvas(client):
    """E10-T17 (reversed 2026-09-14): /loop is the four-step loop, /annotate the
    annotator; the Label step links to the annotate page instead of embedding it."""
    shell = client.get("/loop").get_data(as_text=True)
    for element in ("steps", "panel-select", "panel-label", "panel-train", "panel-review",
                    "btn-open-annotate", "btn-select", "btn-train", "btn-next-round"):
        assert f'id="{element}"' in shell, element
    assert "<iframe" not in shell and "embed=1" not in shell and 'href="/annotate"' in shell
    assert "/loop/rounds" in shell and "/loop/readiness" in shell
    canvas = client.get("/annotate?embed=1&round=1").get_data(as_text=True)
    assert 'id="anno-canvas"' in canvas and 'id="skip-modal"' in canvas
    assert "/images/skip" in canvas and "/similar?" in canvas and "/images/restore" in canvas
    assert client.get("/annotate").get_data(as_text=True) == canvas
    assert client.get("/annotate").get_data(as_text=True) == canvas  # old links
    # every page's nav carries both entries
    for page in ("/", "/annotate", "/loop", "/train", "/evaluate", "/experiments", "/lab"):
        html = client.get(page).get_data(as_text=True)
        assert 'href="/loop" data-tip="Loop"' in html, page
        assert 'href="/annotate" data-tip="Annotate"' in html, page


def test_class_manager_offers_merge_and_train_setup_folds_advanced_knobs(client):
    annotate = client.get("/annotate").get_data(as_text=True)
    assert 'id="merge-row"' in annotate and 'id="merge-target"' in annotate
    assert "/categories/merge" in annotate
    train = client.get("/train").get_data(as_text=True)
    # criterion, seed and the non-primary derived knobs live inside the fold
    details = train[train.index('<details id="advanced-details">'):train.index("</details>")]
    for element in ("criterion-select", "hparams-advanced", "seed-input"):
        assert f'id="{element}"' in details, element
    assert 'id="hparams-list"' in train  # the primary knobs stay in view


def test_every_page_uses_the_shared_controls(client):
    """User decision 2026-09-12: one stepper / checkbox / file-button look on
    every page, served from static/controls.css+js rather than restyled per
    template."""
    for path in PAGES:
        html = client.get(path).get_data(as_text=True)
        assert '/static/controls.css' in html and '/static/controls.js' in html, path
        assert "one checkbox look" not in html, f"{path} restyles checkboxes locally"
    assert client.get("/static/controls.css").status_code == 200
    assert client.get("/static/controls.js").status_code == 200
    # every native number box sits inside a stepper (dynamic ones are built
    # by stepperHTML / horosControls.stepper, which emit the same markup)
    for path in ("/", "/lab", "/train", "/annotate"):
        html = client.get(path).get_data(as_text=True)
        for match in re.finditer(r'<input type="number"[^>]*id="([^"]+)"', html):
            before = html[max(0, match.start() - 60):match.start()]
            assert 'class="stepper' in before, f"{path}: #{match.group(1)} is not a stepper"
    lab = client.get("/lab").get_data(as_text=True)
    assert 'class="file-btn"' in lab and 'id="serve-file"' in lab


def test_lab_serve_offers_every_artifact_format_from_the_capability_list(client):
    """Serve-T3: the Source picker lists ONNX / TensorRT / TFLite / PyTorch /
    checkpoint, and greys the engine out from /api/v1/capabilities (E4-T13),
    never from a hardcoded platform check in the page."""
    lab = client.get("/lab").get_data(as_text=True)
    for fmt in ("onnx", "tensorrt", "tflite", "pytorch", "checkpoint"):
        assert f'["{fmt}",' in lab, fmt
    assert 'api("/capabilities")' in lab and 'unsupported("export_tensorrt")' in lab
    assert "not supported on this platform" in lab


def test_annotator_never_lets_a_previous_image_overwrite_the_current_one(client):
    """E2-T7: every open carries a sequence number; the previous image's late
    load, save response, refetch or assist result is dropped when it lands
    after the user moved on (the user saw stale shapes when paging fast)."""
    html = client.get("/annotate").get_data(as_text=True)
    assert "const seq = this._openSeq = (this._openSeq || 0) + 1;" in html
    assert html.count("if (seq !== this._openSeq) return;") >= 5
    assert "if (this.sam) this._samReset(false, true);" in html  # prompts do not carry over


def test_annotator_deletes_classes_behind_a_blocking_progress_overlay(client):
    html = client.get("/annotate").get_data(as_text=True)
    assert 'id="busy-modal"' in html and 'id="busy-bar"' in html
    assert "/delete`, json(\"POST\", { force })" in html  # the job route, not the sync DELETE
    assert 'window.addEventListener("beforeunload", this._unloadGuard)' in html
    assert "if (this._busyJob) { e.preventDefault(); return; }" in html  # shortcuts off meanwhile


def test_annotator_builds_multi_part_objects_with_plus(client):
    html = client.get("/annotate").get_data(as_text=True)
    assert 'id="sam-part"' in html and 'case "Equal": case "NumpadAdd":' in html
    assert "parts: []" in html and "...(s.parts || []).map((r) => r.flat())" in html
    # Edit: clicking a part makes it the ring with handles
    assert "_activatePartAt(this.editorShapes[sIdx], x, y)" in html


def test_annotator_has_the_sam_tool(client):
    html = client.get("/annotate").get_data(as_text=True)
    assert 'data-tool="sam"' in html and 'id="sam-panel"' in html
    # SAM-T5: Space confirms like Enter (no Next button); several objects are
    # still queued implicitly when a new box is dragged over a live mask, and
    # Clear shows its shortcut
    assert 'id="sam-next"' not in html and "queued: []" in html
    assert 'case "Enter": case "NumpadEnter": case "Space":' in html
    assert "Clear (Esc)" in html
    # a finished shape never gets a made-up class: the page asks instead
    assert 'id="label-modal"' in html and 'id="label-input"' in html
    assert '|| "object"' not in html
    # SAM-T6: boxes are prompts — per shape (⬠ / P), per image, and per class
    # across the project from the Auto-label dialog
    assert 'data-poly="${i}"' in html and 'case "KeyP"' in html
    assert 'id="sam-convert-all"' in html
    for control in ("b2p-class", "b2p-model", "b2p-pending", "b2p-start"):
        assert f'id="{control}"' in html, control
    assert 'api("/segment/boxes"' in html and "/segment`" in html
    for element in ("sam-accept", "sam-clear", "sam-model", "sam-status"):
        assert f'id="{element}"' in html, element
    assert "/segment/prefetch" in html and "/segment`" in html


def test_dataset_page_has_the_danger_zone(client):
    html = client.get("/").get_data(as_text=True)
    assert 'id="danger-panel"' in html and 'id="clear-btn"' in html and 'id="clear-classes"' in html
    assert '"DELETE"' in html and "confirm" in html  # the name-typed confirmation flow


def test_canvas_grid_filters_by_class_and_remembers_the_review_threshold(client):
    """E2-T7 class filter; E3-S2 threshold starts at 0.5 and follows the annotator."""
    html = client.get("/annotate").get_data(as_text=True)
    assert 'id="class-filter"' in html and "All classes" in html
    assert 'q.set("category_id"' in html
    assert 'id="review-threshold" min="0" max="1" step="0.01" value="0.5"' in html
    assert "horos_review_threshold" in html


def test_canvas_class_menu_offers_to_keep_the_class(client):
    """SAM-T4 step 8/8b: accept opens the class menu; a tick skips it next time."""
    html = client.get("/annotate").get_data(as_text=True)
    assert 'id="label-keep"' in html and "horos_keep_class" in html
    assert "_askLabel(hit ? hit.label : typed, hit)" in html
    assert "_suggestClass" in html and 'id="label-hint"' in html  # pseudo label under it wins
    assert 'id="label-suggest"' in html and "horos_suggest_class" in html
    assert "/predictions?threshold=" in html  # no pseudo label: ask the loop's model


def test_dataset_page_groups_photos_and_skips_a_group(client):
    """E10-T20: the Groups fold analyses (embed job, then /images/clusters) and
    skips or restores a whole group through the annotator's skip endpoints."""
    html = client.get("/").get_data(as_text=True)
    assert 'id="groups-fold"' in html and 'id="groups-k"' in html and 'id="groups-btn"' in html
    assert "/images/clusters?" in html and "/loop/embeddings" in html
    assert '"/images/skip"' in html and '"/images/restore"' in html
    assert "skip-unlabeled" in html and "labeled_ids" in html  # skip without / with labeled


def test_annotate_defaults_to_the_to_do_queue_and_offers_the_open_round(client):
    """The plain annotate page opens on "To do"; a round in its Label step is
    prepended as the to-do queue, and the loop page links straight to it."""
    canvas = client.get("/annotate").get_data(as_text=True)
    assert '<option value="unannotated" selected>To do</option>' in canvas
    assert 'cur.state === "labeling"' in canvas and "to label" in canvas
    loop = client.get("/loop").get_data(as_text=True)
    assert 'id="btn-open-annotate"' in loop and "/annotate?round=" in loop


def test_loop_page_offers_training_without_short_classes(client):
    """E10-T8: when only the per-class minimum blocks, a second button trains
    without those classes."""
    html = client.get("/loop").get_data(as_text=True)
    assert 'id="btn-train-without"' in html and "ignore_short_classes" in html
    assert "ready_without_short" in html
