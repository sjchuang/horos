"""E10-T10: a round's images are handed out to annotators; each annotator's
queue is their share plus whatever is unassigned, unlabeled first, on top
of the E2-T8 claims (E10-S7)."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.fake_backend import COLOURS, FakeDetector, FakeEmbedder

from horos.api.annotate import claim_image
from horos.api.loop import assign_round, close_round, round_queue, select_round
from horos.api.project import create_project
from horos.core.dataset import Annotation, Category
from horos.errors import ProjectError


def _project(tmp_path, total=8):
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box")])
    colours = list(COLOURS.values())
    for n in range(1, total + 1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, colours[n % len(colours)])
        project.add_image(path, width=64 + n, height=48)
    return project


def _round(project, count):
    return select_round(project, count=count, embedder=FakeEmbedder(),
                        embedding_model="fake-embedder", detector=FakeDetector(),
                        preannotate=False)


def test_round_robin_assignment_and_per_annotator_queues(tmp_path):
    project = _project(tmp_path)
    record = _round(project, 5)
    assigned = assign_round(project, record.number, ["ann", "bob"])
    owners = [p.assigned_to for p in assigned.selection.picks]
    assert owners.count("ann") == 3 and owners.count("bob") == 2
    assert owners == ["ann", "bob", "ann", "bob", "ann"]  # round robin in pick order

    ann_queue = round_queue(project, record.number, annotator="ann")
    bob_queue = round_queue(project, record.number, annotator="bob")
    assert len(ann_queue) == 3 and len(bob_queue) == 2
    assert {i.image.id for i in ann_queue}.isdisjoint({i.image.id for i in bob_queue})
    assert all(i.assigned_to == "ann" and not i.annotated for i in ann_queue)
    everyone = round_queue(project, record.number)
    assert len(everyone) == 5


def test_assign_keeps_existing_owners_unless_reassigning(tmp_path):
    project = _project(tmp_path)
    record = _round(project, 4)
    assign_round(project, record.number, ["ann"])
    # a third person joins: only unassigned picks would be handed out — none left
    same = assign_round(project, record.number, ["cid"])
    assert all(p.assigned_to == "ann" for p in same.selection.picks)
    redone = assign_round(project, record.number, ["cid", "dee"], reassign=True)
    assert sorted({p.assigned_to for p in redone.selection.picks}) == ["cid", "dee"]


def test_queue_keeps_pick_order_and_shows_claims(tmp_path):
    project = _project(tmp_path)
    record = _round(project, 4)
    first, second = record.image_ids[:2]
    project.save_annotations(
        first, [Annotation(id=1, image_id=first, category_id=1, bbox=(1, 1, 5, 5))],
        expected_version=0,
    )
    claim_image(project, second, "session-x")
    queue = round_queue(project, record.number, session_id="session-y")
    # pick order is kept even for labeled photos: the annotator's position never jumps
    assert [i.image.id for i in queue] == record.image_ids
    assert queue[0].image.id == first and queue[0].annotated
    claimed = next(i for i in queue if i.image.id == second)
    assert claimed.claimed_by == "session-x"
    # the claiming session does not see its own claim as someone else's
    own = round_queue(project, record.number, session_id="session-x")
    assert next(i for i in own if i.image.id == second).claimed_by is None
    # unassigned picks are offered to every annotator
    assert len(round_queue(project, record.number, annotator="zed")) == 4


def test_assignment_validation(tmp_path):
    project = _project(tmp_path)
    record = _round(project, 2)
    with pytest.raises(ProjectError, match="at least one annotator"):
        assign_round(project, record.number, [])
    with pytest.raises(ProjectError, match="distinct"):
        assign_round(project, record.number, ["ann", "ann"])
    with pytest.raises(ProjectError, match="empty"):
        assign_round(project, record.number, ["ann", "  "])
    close_round(project, record.number)
    with pytest.raises(ProjectError, match="closed"):
        assign_round(project, record.number, ["ann"])
