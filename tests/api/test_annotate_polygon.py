"""E2-T3: polygon (instance segmentation) create/edit through the API."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.annotate import save_annotations
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.errors import DatasetValidationError

TRIANGLE = [10.0, 10.0, 30.0, 12.0, 20.0, 28.0]


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _image(project):
    return project.list_images()[0]


def test_polygon_saves_and_derives_bbox(project):
    record = _image(project)
    cat = project.categories[0]
    version = project.load_annotations(record.id).version
    saved = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "segmentation": [TRIANGLE]}],
        expected_version=version,
    )
    ann = saved.annotations[0]
    assert ann.segmentation == [TRIANGLE]
    assert ann.bbox == (10.0, 10.0, 20.0, 18.0)  # derived from polygon extent


def test_polygon_with_explicit_bbox_keeps_it(project):
    record = _image(project)
    cat = project.categories[0]
    version = project.load_annotations(record.id).version
    saved = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "segmentation": [TRIANGLE], "bbox": (9, 9, 22, 20)}],
        expected_version=version,
    )
    assert saved.annotations[0].bbox == (9, 9, 22, 20)


def test_vertex_edit_roundtrip(project):
    record = _image(project)
    cat = project.categories[0]
    version = project.load_annotations(record.id).version
    v1 = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "segmentation": [TRIANGLE]}],
        expected_version=version,
    )
    moved = list(TRIANGLE)
    moved[0], moved[1] = 5.0, 6.0  # drag the first vertex
    v2 = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "segmentation": [moved]}],
        expected_version=v1.version,
    )
    assert v2.annotations[0].segmentation[0][:2] == [5.0, 6.0]
    assert v2.annotations[0].bbox[0] == 5.0  # bbox re-derived


def test_too_few_points_is_rejected(project):
    record = _image(project)
    cat = project.categories[0]
    with pytest.raises(DatasetValidationError, match=">=3"):
        save_annotations(
            project,
            record.id,
            [{"category_id": cat.id, "segmentation": [[1.0, 1.0, 2.0, 2.0]]}],
            expected_version=project.load_annotations(record.id).version,
        )


def test_odd_coordinate_count_is_rejected(project):
    record = _image(project)
    cat = project.categories[0]
    with pytest.raises(DatasetValidationError, match=">=3"):
        save_annotations(
            project,
            record.id,
            [{"category_id": cat.id, "segmentation": [[1.0, 1.0, 2.0, 2.0, 3.0]]}],
            expected_version=project.load_annotations(record.id).version,
        )


def test_multi_part_object_is_one_annotation_whose_bbox_spans_every_ring(project):
    """An occluded object labeled in pieces (+ in the Draw tool) is ONE
    annotation with several polygons; the derived box covers all of them and
    the pieces come back unchanged."""
    record = _image(project)
    rings = [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0],
             [40.0, 30.0, 50.0, 30.0, 50.0, 44.0, 40.0, 44.0]]
    saved = save_annotations(
        project, record.id,
        [{"category_id": project.categories[0].id, "segmentation": rings}],
        expected_version=project.load_annotations(record.id).version,
    )
    ann = saved.annotations[0]
    assert ann.segmentation == rings
    assert ann.bbox == (10.0, 10.0, 40.0, 34.0)
