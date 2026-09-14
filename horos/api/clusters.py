"""Photo groups for the Dataset page (E10-T20): k-means over the project's
embeddings, so a user can look at a handful of representative thumbnails per
group and skip a whole group of unfit photos (duplicates of an empty scene,
a misfiled camera, blank frames) in one action instead of hunting them one
by one in the annotator.

Skipping itself is `images.skip` (E10-T16); this module only proposes the
groups. Groups include labeled and already-skipped photos — the counts are
reported so the page can warn before a skip would drop labels.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from horos.api.embeddings import DEFAULT_EMBEDDING_MODEL, load_embeddings
from horos.api.manifest import capability
from horos.core.clustering import auto_k, kmeans
from horos.core.dataset import ImageRecord
from horos.core.project import Project
from horos.errors import ProjectError

__all__ = ["ImageCluster", "ClusterResult", "cluster_images"]

#: representative thumbnails per group, closest to the centre first
DEFAULT_SAMPLES = 6
MAX_K = 64


class ImageCluster(BaseModel):
    index: int
    size: int
    #: mean cosine similarity of the members to the group centre (1 = identical)
    cohesion: float
    #: the members closest to the centre, the first being the group's medoid
    samples: list[ImageRecord]
    image_ids: list[int]
    #: members that carry confirmed labels — skipping them drops those labels
    #: from training; `labeled_ids` lets a caller skip only the unlabeled rest
    labeled: int = 0
    labeled_ids: list[int] = Field(default_factory=list)
    #: members already skipped
    skipped: int = 0


class ClusterResult(BaseModel):
    model: str
    #: groups actually formed (k-means may leave fewer than asked)
    k: int
    requested_k: int | None = None
    seed: int = 0
    total_images: int
    #: images with a current embedding — the ones that were grouped
    embedded: int
    clusters: list[ImageCluster] = Field(default_factory=list)


@capability(
    "images.clusters",
    summary="Group the project's photos by embedding similarity (k-means)",
    web_route="/api/v1/images/clusters",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Drives the Dataset page's photo groups; scripts call the Python API.",
)
def cluster_images(
    project: Project,
    *,
    k: int | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> ClusterResult:
    """Group every photo with a current `model` embedding into `k` groups
    (None = automatic). Photos without an embedding are left out and
    counted; with none embedded at all the answer is an explicit error so
    the caller runs the embedding job first (E10-T3)."""
    if k is not None and not 1 <= k <= MAX_K:
        raise ProjectError(f"k must be within [1, {MAX_K}], got {k}")
    if samples < 1:
        raise ProjectError(f"samples must be at least 1, got {samples}")
    records = project.list_images()
    embedded: list[ImageRecord] = []
    vectors = []
    all_vecs = load_embeddings(project, [r.id for r in records], model)
    if all_vecs is not None:
        embedded, vectors = list(records), [all_vecs[i] for i in range(len(records))]
    else:
        for record in records:  # some lack a vector: group the ones that have one
            vec = load_embeddings(project, [record.id], model)
            if vec is not None:
                embedded.append(record)
                vectors.append(vec[0])
    if not embedded:
        raise ProjectError(
            f"No photo has a current {model} embedding yet — run the embedding job "
            f"(POST /loop/embeddings) first."
        )
    matrix = np.stack(vectors)
    wanted = k if k is not None else auto_k(len(embedded))
    groups = kmeans(matrix, wanted, seed=seed)
    labeled = {
        r.id for r in embedded
        if any(a.status == "confirmed" for a in project.load_annotations(r.id).annotations)
    }
    clusters = [
        ImageCluster(
            index=i,
            size=len(g.members),
            cohesion=g.cohesion,
            samples=[embedded[m] for m in g.members[:samples]],
            image_ids=[embedded[m].id for m in g.members],
            labeled=sum(embedded[m].id in labeled for m in g.members),
            labeled_ids=[embedded[m].id for m in g.members if embedded[m].id in labeled],
            skipped=sum(embedded[m].excluded for m in g.members),
        )
        for i, g in enumerate(groups)
    ]
    return ClusterResult(
        model=model, k=len(clusters), requested_k=k, seed=seed,
        total_images=len(records), embedded=len(embedded), clusters=clusters,
    )
