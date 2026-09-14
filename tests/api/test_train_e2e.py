"""E5-T7 (hard acceptance): a 20-image fixture dataset trains for 2 epochs
through the real worker subprocess and produces a loadable checkpoint.

Runs only where the training stack is installed (rfdetr[train] — pulled in by
`pip install horos` on non-Jetson platforms, or `horos doctor --fix`); skipped
otherwise so the rest of the suite stays ML-free.
"""

from __future__ import annotations

import importlib.util
import json
import random
import time
from pathlib import Path

import pytest

_MISSING = [
    name
    for name in ("torch", "rfdetr", "pytorch_lightning", "albumentations")
    if importlib.util.find_spec(name) is None
]
pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"training stack not installed: {', '.join(_MISSING)}"
)

IMAGES = 20  # 16 train / 4 valid
SIZE = 128


def _make_fixture_coco(root: Path) -> Path:
    """20 synthetic images, one bright box on a dark background per image."""
    from PIL import Image, ImageDraw

    rng = random.Random(42)
    categories = [{"id": 1, "name": "block", "supercategory": "none"}]
    for split, count, offset in (("train", 16, 0), ("valid", 4, 16)):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        images, annotations = [], []
        for i in range(count):
            image_id = offset + i + 1
            name = f"img_{image_id:03d}.png"
            x = rng.randint(8, SIZE - 48)
            y = rng.randint(8, SIZE - 48)
            w = rng.randint(24, 40)
            h = rng.randint(24, 40)
            canvas = Image.new("RGB", (SIZE, SIZE), (16, 16, 24))
            ImageDraw.Draw(canvas).rectangle(
                (x, y, x + w, y + h), fill=(240, 200, 40)
            )
            canvas.save(split_dir / name)
            images.append(
                {"id": image_id, "file_name": name, "width": SIZE, "height": SIZE}
            )
            annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(
                {"images": images, "annotations": annotations, "categories": categories}
            ),
            "utf-8",
        )
    return root


def test_small_dataset_trains_to_a_loadable_checkpoint(tmp_path):
    from horos.api import create_project, import_dataset
    from horos.api.train import TrainRunConfig, start_training, training_status

    project = create_project(tmp_path / "proj")
    import_dataset(project, _make_fixture_coco(tmp_path / "coco"))

    record = start_training(
        project,
        TrainRunConfig(
            model="rfdetr-nano",
            epochs=2,
            batch_size=2,
            seed=42,
            # single-process data loading: deterministic and spawn-trivial in CI
            extra={"num_workers": 0, "grad_accum_steps": 1, "multi_scale": False},
        ),
    )

    deadline = time.monotonic() + 1800  # includes the one-off nano weights download
    while time.monotonic() < deadline:
        status = training_status(project, record.run_id)
        if status.run.state not in ("pending", "running"):
            break
        time.sleep(5)
    else:
        pytest.fail("training did not finish within 30 minutes")

    assert status.run.state == "completed", status.run.error
    checkpoint = Path(status.run.checkpoint)
    assert checkpoint.is_file() and checkpoint.stat().st_size > 1_000_000

    types = {e["type"] for e in status.events}
    assert {"started", "progress", "metrics", "completed"} <= types

    # E5-T7's bar is a *loadable* checkpoint: load it and run one inference.
    from horos.backends import get_backend

    backend = get_backend("rfdetr-nano", checkpoint=checkpoint)
    sample = next((project.root / "runs" / record.run_id / "dataset" / "valid").glob("*.png"))
    prediction = backend.infer_one(sample, threshold=0.1)
    assert prediction.width == SIZE and prediction.height == SIZE


def test_warm_start_continues_with_a_new_class(tmp_path):
    """E5-S6 / E10-T8: a second run starts from the first run's best weights
    (init_from) while the project has gained a class. rfdetr resizes the
    class head on load, so the run completes and its snapshot names both
    classes — the loop relies on this to keep training as users add classes."""
    from horos.api import create_project, import_dataset
    from horos.api.train import (
        TrainRunConfig,
        _snapshot_class_names,
        start_training,
        training_status,
    )
    from horos.core.dataset import Annotation, Category

    project = create_project(tmp_path / "proj")
    import_dataset(project, _make_fixture_coco(tmp_path / "coco"))
    quick = dict(batch_size=2, seed=42,
                 extra={"num_workers": 0, "grad_accum_steps": 1, "multi_scale": False})

    def _run(config):
        record = start_training(project, config)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            status = training_status(project, record.run_id)
            if status.run.state not in ("pending", "running"):
                return status.run
            time.sleep(5)
        pytest.fail("training did not finish within 30 minutes")

    first = _run(TrainRunConfig(model="rfdetr-nano", epochs=1, **quick))
    assert first.state == "completed", first.error
    assert _snapshot_class_names(Path(first.checkpoint)) == ["block"]

    # a new class arrives with a few labels on training photos
    project.set_categories([*project.categories, Category(id=2, name="extra")])
    for record in project.list_images():
        if record.split != "train":
            continue
        stored = project.load_annotations(record.id)
        anns = [*stored.annotations,
                Annotation(id=max((a.id for a in stored.annotations), default=0) + 1,
                           image_id=record.id, category_id=2, bbox=(2.0, 2.0, 12.0, 12.0))]
        project.save_annotations(record.id, anns, expected_version=stored.version)

    second = _run(TrainRunConfig(model="rfdetr-nano", epochs=1, init_from=first.checkpoint,
                                 **quick))
    assert second.state == "completed", second.error
    assert second.config["init_from"] == first.checkpoint
    assert _snapshot_class_names(Path(second.checkpoint)) == ["block", "extra"]
    # the head itself grew: 2 classes + background (rfdetr keeps the checkpoint's
    # head size unless told the new count — demo_project died with a CUDA
    # device-side assert when a 2-class head met 7 classes)
    import torch

    def head_rows(checkpoint):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        return state["class_embed.weight"].shape[0]

    assert head_rows(first.checkpoint) == 2 and head_rows(second.checkpoint) == 3

    from horos.backends import get_backend

    backend = get_backend("rfdetr-nano", checkpoint=Path(second.checkpoint))
    sample = next((project.root / "runs" / second.run_id / "dataset" / "valid").glob("*.png"))
    prediction = backend.infer_one(sample, threshold=0.05)
    assert prediction.width == SIZE
