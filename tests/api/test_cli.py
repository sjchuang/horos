"""E9-T5: the CLI runs the whole E1 workflow without a browser (E9-S3)."""

import json
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.cli import main


def _run(capsys, *argv) -> tuple[int, dict | list]:
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip() else None)


def test_full_workflow(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    proj = tmp_path / "proj"

    code, body = _run(capsys, "init", str(proj), "--name", "demo")
    assert code == 0 and body["name"] == "demo"

    code, body = _run(capsys, "import", str(coco_dir), "--project", str(proj))
    assert code == 0 and body["num_images"] == 3

    code, body = _run(capsys, "stats", "--project", str(proj))
    assert code == 0 and body["num_annotations"] == 4

    code, body = _run(capsys, "split", "--project", str(proj), "--reshuffle",
                      "--train", "1.0", "--valid", "0.0", "--test", "0.0")
    assert code == 0 and body["train"] == 3

    code, body = _run(capsys, "export", str(tmp_path / "out"),
                      "--project", str(proj), "--format", "yolo")
    assert code == 0 and body["path"].endswith("data.yaml")


def test_validate_exit_code_reflects_dataset_health(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    proj = tmp_path / "proj"
    _run(capsys, "init", str(proj))
    _run(capsys, "import", str(coco_dir), "--project", str(proj))

    code, body = _run(capsys, "validate", "--project", str(proj))
    assert code == 0 and body["ok"] is True

    # break it: delete an image file
    from horos.api import open_project

    project = open_project(proj)
    (project.images_dir / project.list_images()[0].file_name).unlink()
    code, body = _run(capsys, "validate", "--project", str(proj))
    assert code == 1 and body["ok"] is False


def test_convert(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    code, body = _run(capsys, "convert", str(coco_dir), str(tmp_path / "yolo"),
                      "--to", "yolo")
    assert code == 0 and body["path"].endswith("data.yaml")


def test_convert_to_labelme(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    out = tmp_path / "labelme"
    code, body = _run(capsys, "convert", str(coco_dir), str(out), "--to", "labelme")
    assert code == 0 and body["path"] == str(out)
    assert (out / "train" / "a.json").is_file() and (out / "valid" / "c.json").is_file()


def test_catalog_lists_architectures_with_licenses(capsys):
    code, body = _run(capsys, "catalog")
    assert code == 0
    assert all(m["weights_license"] == "Apache-2.0" for m in body)


def test_models_lists_the_projects_trained_models(tmp_path, monkeypatch, capsys):
    from helpers.runs import completed_fake_run

    project, record = completed_fake_run(tmp_path, epochs=2)
    monkeypatch.chdir(project.root)
    code, body = _run(capsys, "models")
    assert code == 0 and [m["run_id"] for m in body] == [record.run_id]
    entry = body[0]
    assert entry["state"] == "completed" and entry["default"] is True
    assert entry["classes"] == ["forklift", "pallet"] and entry["epochs_completed"] == 2
    assert entry["scores"]["loss"] == pytest.approx(0.5)
    assert entry["checkpoint"].endswith("best.fake")


def test_models_without_all_hides_unfinished_runs(tmp_path, monkeypatch, capsys):
    import time

    from helpers.runs import FAKE, ensure_worker_can_import_helpers

    from horos.api import create_project, import_dataset
    from horos.api.train import TrainRunConfig, start_training, training_status

    ensure_worker_can_import_helpers()
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    failed = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=1, extra={"fail": True})
    )
    deadline = time.monotonic() + 60
    while training_status(project, failed.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    monkeypatch.chdir(project.root)
    code, body = _run(capsys, "models")
    assert code == 0 and body == []
    code, body = _run(capsys, "models", "--all")
    assert code == 0 and [m["state"] for m in body] == ["failed"]
    assert body[0]["default"] is False


def test_capabilities(capsys):
    code, body = _run(capsys, "capabilities")
    assert code == 0
    assert {f["feature"] for f in body["features"]} >= {"training", "export_tensorrt"}


def test_horos_errors_exit_2_with_stderr(tmp_path, capsys):
    code = main(["stats", "--project", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert code == 2
    assert "error [project_error]" in captured.err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_train_streams_events_and_exits_by_state(tmp_path, capsys, monkeypatch):
    """`horos train` runs in the foreground: it starts a run, prints the event
    stream as JSONL, and its exit code mirrors the terminal state (E5/E9-S3)."""
    import os

    from helpers.data import write_sample_coco_dir as _make

    import horos.api as api
    from horos.api.train import TrainRunConfig
    from horos.cli import main as cli_main

    tests_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", tests_root + (os.pathsep + existing if existing else "")
    )

    proj_dir = tmp_path / "proj"
    project = api.create_project(proj_dir)
    api.import_dataset(project, _make(tmp_path / "coco"))

    # the CLI builds the config itself; reroute it onto the fake backend
    original = api.start_training

    def with_fake(project, config):
        patched = TrainRunConfig(
            **config.model_dump(exclude={"entrypoint_override"}),
            entrypoint_override="helpers.fake_backend:FakeBackend",
        )
        return original(project, patched)

    monkeypatch.setattr(api, "start_training", with_fake)
    # the CLI refuses ML commands when torch/rfdetr are absent; the fake backend
    # needs no torch, so bypass the gate instead of tying this test to what is
    # installed (the default setup_local.sh venv has no ML stack)
    import horos.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_ml_preflight", lambda command: None)

    code = cli_main(["train", "--project", str(proj_dir), "--epochs", "2"])
    out = capsys.readouterr().out
    # events stream as one-line JSON; the final run record is pretty-printed —
    # decode the concatenated stream object by object
    decoder, pos, payloads = json.JSONDecoder(), 0, []
    while pos < len(out):
        remainder = out[pos:].lstrip()
        if not remainder:
            break
        obj, consumed = decoder.raw_decode(remainder)
        payloads.append(obj)
        pos += (len(out[pos:]) - len(remainder)) + consumed
    assert code == 0
    types = [p.get("type") for p in payloads]
    assert "started" in types and "completed" in types
    # the last JSON payload is the final run record
    assert payloads[-1]["state"] == "completed"


# ------------------------------------------------------- project & run discovery


def test_init_in_an_empty_directory_uses_it_directly(tmp_path, monkeypatch, capsys):
    """`horos init <name>` in an empty directory makes THAT directory the
    project — no pointless nesting (the common `mkdir x && cd x` flow)."""
    empty = tmp_path / "beds"
    empty.mkdir()
    monkeypatch.chdir(empty)
    code, body = _run(capsys, "init", "beds")
    assert code == 0
    assert Path(body["root"]) == empty.resolve() and body["name"] == "beds"
    assert (empty / "horos.json").is_file()


def test_init_ignores_dotfiles_when_deciding_emptiness(tmp_path, monkeypatch, capsys):
    # a fresh `git init` must not push the project into a subdirectory
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    monkeypatch.chdir(root)
    code, body = _run(capsys, "init", "repo")
    assert code == 0 and Path(body["root"]) == root.resolve()


def test_init_with_files_present_creates_a_subdirectory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("keep me", encoding="utf-8")
    monkeypatch.chdir(root)
    code, body = _run(capsys, "init", "proj")
    assert code == 0 and Path(body["root"]) == (root / "proj").resolve()
    assert (root / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_init_without_a_name_uses_the_current_directory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "unnamed"
    root.mkdir()
    monkeypatch.chdir(root)
    code, body = _run(capsys, "init")
    assert code == 0 and body["name"] == "unnamed"


def test_init_with_a_path_still_creates_that_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, body = _run(capsys, "init", "nested/deep/proj")
    assert code == 0 and Path(body["root"]).resolve() == (tmp_path / "nested/deep/proj").resolve()


def test_project_commands_find_the_project_from_a_subdirectory(tmp_path, monkeypatch, capsys):
    from horos.api import create_project, import_dataset

    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    monkeypatch.chdir(project.images_dir)  # a subdirectory of the project
    code, body = _run(capsys, "stats")
    assert code == 0 and body["num_images"] == 3


def test_missing_project_names_the_two_ways_out(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["stats"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--project" in captured.err and "horos init" in captured.err


def test_run_defaults_to_the_newest_completed_run(tmp_path, monkeypatch, capsys):
    from helpers.runs import completed_fake_run

    project, record = completed_fake_run(tmp_path, epochs=1)
    monkeypatch.chdir(project.root)
    code = main(["report", "--format", "xlsx"])
    captured = capsys.readouterr()  # one read: it drains both streams
    assert code == 0 and record.run_id in json.loads(captured.out)["path"]
    # the choice is announced on stderr, so stdout stays machine-readable
    assert f"using run {record.run_id}" in captured.err


def test_run_default_without_any_run_explains_itself(tmp_path, monkeypatch, capsys):
    from horos.api import create_project

    project = create_project(tmp_path / "empty_proj")
    monkeypatch.chdir(project.root)
    code = main(["report"])
    assert code == 2 and "no training runs yet" in capsys.readouterr().err


def test_analyze_prints_the_analysis_and_worst_images(tmp_path, monkeypatch, capsys):
    from helpers.runs import completed_fake_run

    from horos.api.evaluate import _write_detections

    project, record = completed_fake_run(tmp_path, epochs=1)
    monkeypatch.chdir(project.root)
    code = main(["analyze", "--split", "train"])
    assert code == 2  # no evaluation yet -> horos error, explained on stderr
    assert "run an evaluation first" in capsys.readouterr().err

    _write_detections(project, record.run_id, "train", [])  # every gt box missed
    code, body = _run(capsys, "analyze", "--split", "train", "--worst", "1", "--threshold", "0.3")
    assert code == 0
    assert body["analysis"]["threshold"] == 0.3 and body["analysis"]["fn"] == 3
    assert body["worst"]["total_images"] == 2 and len(body["worst"]["images"]) == 1

    code, body = _run(capsys, "analyze", "--split", "train", "--worst", "0")
    assert code == 0 and "worst" not in body


def test_analyze_overlays_writes_one_png_per_worst_image(tmp_path, monkeypatch, capsys):
    from helpers.runs import completed_fake_run

    from horos.api.evaluate import _write_detections

    project, record = completed_fake_run(tmp_path, epochs=1)
    monkeypatch.chdir(project.root)
    _write_detections(project, record.run_id, "train", [])
    out_dir = tmp_path / "overlays"
    code, body = _run(
        capsys, "analyze", "--split", "train", "--worst", "5", "--overlays", str(out_dir)
    )
    assert code == 0
    assert sorted(Path(p).name for p in body["overlays"]) == ["a.overlay.png", "b.overlay.png"]
    assert all(Path(p).is_file() for p in body["overlays"])


def test_infer_overlay_dir_writes_the_drawn_image(tmp_path, monkeypatch, capsys):
    from helpers.data import make_image
    from helpers.runs import completed_fake_run

    project, record = completed_fake_run(tmp_path, epochs=1)
    monkeypatch.chdir(project.root)
    import horos.cli as cli_mod

    # the fake backend needs no ML stack: bypass the pre-flight like the
    # other CLI tests do, so this runs on the torch-free CI matrix too
    monkeypatch.setattr(cli_mod, "_ml_preflight", lambda command: None)
    probe = make_image(tmp_path / "probe.png", 64, 48)
    code = main(["infer", str(probe), "--overlay-dir", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out" / "probe.overlay.png").is_file()
    assert json.loads(capsys.readouterr().out)["instances"][0]["score"] == 0.9


def test_install_needs_no_gpu_flag_on_an_amd_machine(capsys, monkeypatch):
    """`horos install` alone must reach the ROCm wheels (E4/§4).

    There is deliberately no --rocm flag: an AMD GPU is handled like an
    NVIDIA one. Driven through --dry-run, so it also proves the CLI leaves
    the decision to the planner.
    """
    from horos.api import install as install_mod
    from horos.core.platform_info import PlatformInfo

    # Stand in for the platform too: plan_install() drops the AMD GPU on
    # macOS and Jetson (correctly — ROCm has no build there), so without
    # this the ROCm path is unreachable on a Mac and this test would only
    # pass on the Linux/Windows CI runners.
    monkeypatch.setattr(
        install_mod,
        "detect_platform",
        lambda: PlatformInfo(
            os_family="linux", arch="x86_64", is_jetson=False,
            python_version="3.10.6",
        ),
    )
    monkeypatch.setattr(install_mod, "detect_amd_gpu", lambda: "AMD Radeon RX 9070 XT")
    monkeypatch.setattr(install_mod, "detect_rocm_arch", lambda: "gfx1201")
    monkeypatch.setattr(install_mod, "detect_cuda_version", lambda: None)
    monkeypatch.setattr(install_mod, "probe_missing", lambda *a, **k: ["torch"])

    assert main(["install", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "torch[device-gfx1201]" in out
    assert "stable.repo.amd.com" in out

    # --cpu is the opt-out
    assert main(["install", "--cpu", "--dry-run"]) == 0
    cpu_out = capsys.readouterr().out
    assert "stable.repo.amd.com" not in cpu_out

    # and the flag really is gone
    with pytest.raises(SystemExit):
        main(["install", "--rocm", "gfx1201", "--dry-run"])


def test_boxes_to_polygons_streams_events_and_filters_by_class(tmp_path, monkeypatch, capsys):
    """SAM-T6 from the CLI: JSONL events like autolabel; --class narrows it."""
    import json

    from helpers.data import write_sample_coco_dir
    from helpers.fake_backend import FakePromptableSegmenter

    import horos.backends
    import horos.cli as cli_mod
    from horos.api.dataset import import_dataset
    from horos.api.project import create_project
    from horos.api.segment import _reset_segmenters

    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    fake = FakePromptableSegmenter()
    monkeypatch.setattr(horos.backends, "get_backend", lambda key, **kw: fake)
    monkeypatch.setattr(cli_mod, "_ml_preflight", lambda command: None)
    _reset_segmenters()
    try:
        name = project.categories[0].name
        code = main(["boxes-to-polygons", "--project", str(project.root), "--class", name])
        assert code == 0
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
        assert events[0]["type"] == "started" and events[-1]["type"] == "completed"
        assert events[0]["config"]["categories"] == [project.categories[0].id]
        assert events[-1]["result"]["converted"] >= 1
        # a bad class name fails through the stream, exit code 2
        code = main(["boxes-to-polygons", "--project", str(project.root), "--class", "ghost"])
        assert code == 2
        assert json.loads(capsys.readouterr().out.splitlines()[-1])["type"] == "failed"
    finally:
        _reset_segmenters()


def test_loop_status_and_select_from_the_cli(tmp_path, capsys, monkeypatch):
    """E10-T15: the loop runs from the CLI without a browser (E9-S3)."""
    from helpers.fake_backend import fake_get_backend

    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    proj = tmp_path / "proj"
    _run(capsys, "init", str(proj), "--name", "demo")
    _run(capsys, "import", str(coco_dir), "--project", str(proj))
    # the sample dataset is fully labeled; wipe one image's labels to get a pool
    from horos.api import open_project

    project = open_project(proj)
    stored = project.load_annotations(1)
    project.save_annotations(1, [], expected_version=stored.version)

    code, body = _run(capsys, "loop", "--project", str(proj))
    assert code == 0 and body["pool_size"] == 1 and body["labeled_images"] == 2
    assert body["next_strategy"] == "pal" and body["rounds"] == []

    monkeypatch.setattr("horos.backends.get_backend", fake_get_backend)
    # 'loop select' is ML-gated (it needs the embedding / scoring models); the
    # torch-free CI must still exercise the command, so the gate is bypassed
    # here and the fakes stand in for the models
    monkeypatch.setattr("horos.cli._ml_preflight", lambda command: None)
    code, body = _run(capsys, "loop", "select", "--project", str(proj),
                      "--count", "1", "--strategy", "diversity")
    assert code == 0 and body["number"] == 1 and body["state"] == "labeling"
    assert [p["image_id"] for p in body["selection"]["picks"]] == [1]
    assert body["selection"]["strategy"] == "diversity"

    code, body = _run(capsys, "loop", "--project", str(proj))
    assert code == 0 and body["current"]["number"] == 1 and body["pool_size"] == 0

    # 'train' is refused with the readiness reasons (3 labeled images), not a trace
    code, body = _run(capsys, "loop", "train", "--project", str(proj))
    assert code != 0 and body is None
    code, body = _run(capsys, "loop", "close", "--project", str(proj))
    assert code == 0 and body["state"] == "closed"
    code, body = _run(capsys, "loop", "close", "--project", str(proj))
    assert code != 0  # nothing open any more


def test_import_photos_and_the_label_policy_flag(tmp_path, capsys):
    # E1-T11: a photo directory imports unlabeled; --on-annotations reaches the API
    import json

    from helpers.data import make_image, write_sample_coco_dir

    from horos.api import open_project
    from horos.core.dataset import Annotation

    proj = tmp_path / "proj"
    main(["init", str(proj)])
    capsys.readouterr()
    photos = tmp_path / "photos"
    make_image(photos / "a.png", 64, 48)
    make_image(photos / "b.png", 64, 48)
    code = main(["import", str(photos), "--project", str(proj)])
    body = json.loads(capsys.readouterr().out)
    assert code == 0 and body["format"] == "images" and body["num_images"] == 2
    project = open_project(proj)
    record = next(r for r in project.list_images() if r.file_name == "a.png")
    project.save_annotations(
        record.id,
        [Annotation(id=1, image_id=record.id, category_id=1, bbox=(9.0, 9.0, 3.0, 3.0))],
        expected_version=0,
    )
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    code = main(["import", str(coco_dir), "--project", str(proj), "--on-annotations", "skip"])
    body = json.loads(capsys.readouterr().out)
    assert code == 0 and body["annotations_kept"] == 1 and body["images_matched"] == 2
