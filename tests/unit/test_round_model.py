"""E10-T1: round data model, state machine and storage."""

import pytest

from horos.core.project import Project
from horos.core.rounds import (
    LoopRound,
    PickedImage,
    SelectionRecord,
    create_round,
    current_round,
    list_rounds,
    load_round,
    round_dir,
    save_round,
)
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / "proj", name="loop")


def _selection(*ids):
    return SelectionRecord(
        strategy="diversity",
        requested=len(ids),
        pool_size=10,
        embedding_model="dinov2-small",
        picks=[PickedImage(image_id=i, score=0.5, reason="test") for i in ids],
    )


def test_fresh_project_has_no_rounds(project):
    assert list_rounds(project) == []
    assert current_round(project) is None


def test_create_round_numbers_sequentially_and_persists(project):
    first = create_round(project, labeled_before=0)
    assert first.number == 1 and first.state == "selecting"
    assert (round_dir(project, 1) / "round.json").is_file()
    assert current_round(project).number == 1

    save_round(project, first.advance("closed"))
    second = create_round(project, labeled_before=20)
    assert second.number == 2 and second.labeled_before == 20
    assert [r.number for r in list_rounds(project)] == [1, 2]


def test_only_one_open_round_at_a_time(project):
    create_round(project, labeled_before=0)
    with pytest.raises(ProjectError, match="still 'selecting'"):
        create_round(project, labeled_before=0)


def test_selection_roundtrips_with_scores_and_reasons(project):
    record = create_round(project, labeled_before=0)
    record = record.model_copy(update={"selection": _selection(3, 7)}).advance("labeling")
    save_round(project, record)
    loaded = load_round(project, 1)
    assert loaded.state == "labeling"
    assert loaded.image_ids == [3, 7]
    assert loaded.selection.picks[0].reason == "test"
    assert loaded.selection.strategy == "diversity"


def test_state_machine_follows_the_loop():
    r = LoopRound(number=1)
    r = r.advance("labeling").advance("training").advance("reviewing")
    assert r.closed_at is None
    r = r.advance("closed")
    assert r.state == "closed" and r.closed_at is not None
    with pytest.raises(ProjectError, match="cannot go from 'closed'"):
        r.advance("labeling")


def test_failed_training_can_fall_back_to_labeling():
    r = LoopRound(number=1).advance("labeling").advance("training")
    assert r.advance("labeling").state == "labeling"


def test_illegal_skip_is_refused():
    with pytest.raises(ProjectError, match="allowed: labeling, closed"):
        LoopRound(number=1).advance("training")


def test_missing_and_corrupt_rounds_are_explicit(project):
    with pytest.raises(ProjectError, match="No round 4"):
        load_round(project, 4)
    create_round(project, labeled_before=0)
    (round_dir(project, 1) / "round.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(ProjectError, match="Corrupt round record"):
        load_round(project, 1)


def test_stray_directories_under_rounds_are_ignored(project):
    create_round(project, labeled_before=0)
    (project.root / "rounds" / "notes").mkdir()
    (project.root / "rounds" / "9").mkdir()  # no round.json inside
    assert [r.number for r in list_rounds(project)] == [1]
