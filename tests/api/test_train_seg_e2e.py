"""E5-T10: a polygon fixture trains RF-DETR-Seg Nano for 2 epochs through the
real worker and the resulting checkpoint returns polygons at inference.

Runs only where the training stack is installed (see test_train_e2e.py)."""

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

SIZE = 160


def _make_polygon_coco(root: Path) -> Path:
    """20 synthetic images with one bright diamond each, annotated as a polygon."""
    from PIL import Image, ImageDraw

    rng = random.Random(7)
    categories = [{"id": 1, "name": "diamond", "supercategory": "none"}]
    for split, count, offset in (("train", 16, 0), ("valid", 4, 16)):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        images, annotations = [], []
        for i in range(count):
            image_id = offset + i + 1
            name = f"img_{image_id:03d}.png"
            cx, cy = rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40)
            r = rng.randint(18, 30)
            pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            canvas = Image.new("RGB", (SIZE, SIZE), (18, 20, 28))
            ImageDraw.Draw(canvas).polygon(pts, fill=(240, 200, 40))
            canvas.save(split_dir / name)
            images.append({"id": image_id, "file_name": name, "width": SIZE, "height": SIZE})
            annotations.append({
                "id": image_id, "image_id": image_id, "category_id": 1,
                "bbox": [cx - r, cy - r, 2 * r, 2 * r], "area": 2 * r * r, "iscrowd": 0,
                "segmentation": [[float(v) for p in pts for v in p]],
            })
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": categories}),
            "utf-8",
        )
    return root


def test_polygon_dataset_trains_a_segmentation_model_that_returns_polygons(tmp_path):
    from horos.api import create_project, import_dataset
    from horos.api.train import TrainRunConfig, start_training, training_status

    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, _make_polygon_coco(tmp_path / "coco"))
    assert summary.num_annotations == 20

    record = start_training(
        project,
        TrainRunConfig(
            model="rfdetr-seg-nano",
            epochs=2,
            batch_size=2,
            seed=42,
            extra={"num_workers": 0, "grad_accum_steps": 1, "multi_scale": False},
        ),
    )
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        status = training_status(project, record.run_id)
        if status.run.state not in ("pending", "running"):
            break
        time.sleep(5)
    else:
        pytest.fail("segmentation training did not finish within 30 minutes")

    assert status.run.state == "completed", status.run.error
    checkpoint = Path(status.run.checkpoint)
    assert checkpoint.is_file() and checkpoint.stat().st_size > 1_000_000
    assert status.run.model == "rfdetr-seg-nano"

    from horos.backends import get_backend

    backend = get_backend("rfdetr-seg-nano", checkpoint=checkpoint)
    sample = next((project.root / "runs" / record.run_id / "dataset" / "valid").glob("*.png"))
    prediction = backend.infer_one(sample, threshold=0.05)
    assert prediction.width == SIZE and prediction.height == SIZE
    # after two epochs the mask head is not trustworthy, but every instance a
    # segmentation model reports must carry a polygon, never a bare box
    for inst in prediction.instances:
        assert inst.segmentation and len(inst.segmentation[0]) >= 6
    assert all(c.segmentation is None for c in prediction.candidates)
