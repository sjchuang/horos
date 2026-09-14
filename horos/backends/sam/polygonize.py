"""Binary mask -> simplified polygon, dependency-free.

The project deliberately has no OpenCV dependency, so the two classic pieces
are implemented directly: Moore-neighbor boundary tracing (with Jacob's
stopping criterion) and Douglas-Peucker simplification. Pixel coordinates use
cell centers, so a polygon vertex (x, y) sits on pixel (row y, col x).

The mask may be any 2D indexable (numpy array or nested lists). A prompted
mask is rarely one clean blob: noisy images give SAM stray islands of a few
pixels, and a box prompt on a cluttered scene picks up crumbs of the
neighbours. Only the LARGEST 8-connected blob is kept — traced as the polygon
and measured for the box — so the polygon, its box and its area always
describe the same pixels. (Tracing whichever blob held the top-most pixel
turned a stray speck into the "polygon" of an otherwise correct box.)
"""

from __future__ import annotations

from typing import NamedTuple

# Moore neighborhood, clockwise, starting west: (dx, dy)
_MOORE = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


class MaskShape(NamedTuple):
    """The largest blob of a mask: its simplified outline, COCO xywh box in
    pixel counts, and pixel area."""

    polygon: list[float]
    bbox: tuple[float, float, float, float]
    area: int


def _row_runs(row, width: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) foreground runs of one mask row."""
    if hasattr(row, "shape"):  # numpy fast path
        import numpy as np

        padded = np.concatenate(([False], np.asarray(row, dtype=bool), [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1]).tolist()
        return list(zip(edges[0::2], edges[1::2], strict=True))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x in range(width):
        if row[x]:
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, width))
    return runs


def largest_blob(
    mask, anchor: tuple[int, int] | None = None
) -> tuple[list[tuple[int, int, int]], int, tuple[int, int, int, int]] | None:
    """The largest 8-connected foreground component as (runs, area, box) —
    or, with `anchor` = (x, y) on a foreground pixel, the component under
    that pixel: the piece the annotator clicked, even when the model's mask
    also covers a bigger piece of the same object elsewhere.

    `runs` are (y, start, stop) half-open row runs; `box` is (x0, y0, x1, y1)
    in half-open pixel coordinates. None for an empty mask. Row runs are
    unioned with an overlap test against the previous row, so the work is
    proportional to the number of runs, not pixels.
    """
    height = len(mask)
    width = len(mask[0]) if height else 0
    parent: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    runs: list[tuple[int, int, int]] = []  # (y, start, stop) indexed like parent
    previous: list[int] = []  # run indices of the row above
    for y in range(height):
        current: list[int] = []
        for start, stop in _row_runs(mask[y], width):
            index = len(runs)
            runs.append((y, start, stop))
            parent.append(index)
            current.append(index)
            for above in previous:
                _, a0, a1 = runs[above]
                if a0 <= stop and start <= a1:  # touching, corners included (8-conn)
                    union(index, above)
        previous = current
    if not runs:
        return None
    members: dict[int, list[int]] = {}
    for index in range(len(runs)):
        members.setdefault(find(index), []).append(index)
    best = None
    if anchor is not None:
        ax, ay = int(anchor[0]), int(anchor[1])
        hit = next((i for i, (y, start, stop) in enumerate(runs)
                    if y == ay and start <= ax < stop), None)
        if hit is not None:
            best = members[find(hit)]
    if best is None:
        best = max(members.values(), key=lambda idx: sum(runs[i][2] - runs[i][1] for i in idx))
    blob = [runs[i] for i in best]
    area = sum(stop - start for _, start, stop in blob)
    x0 = min(start for _, start, _ in blob)
    x1 = max(stop for _, _, stop in blob)
    y0 = min(y for y, _, _ in blob)
    y1 = max(y for y, _, _ in blob) + 1
    return blob, area, (x0, y0, x1, y1)


def _blob_mask(blob, height: int, width: int):
    """A nested-list mask holding only the blob's pixels."""
    rows = [bytearray(width) for _ in range(height)]
    for y, start, stop in blob:
        rows[y][start:stop] = b"\x01" * (stop - start)
    return rows


def _first_foreground(mask, height: int, width: int) -> tuple[int, int] | None:
    any_row = getattr(mask, "any", None)
    if any_row is not None and hasattr(mask, "shape"):  # numpy fast path
        rows = mask.any(axis=1)
        for y in range(height):
            if rows[y]:
                row = mask[y]
                for x in range(width):
                    if row[x]:
                        return x, y
        return None
    for y in range(height):
        row = mask[y]
        for x in range(width):
            if row[x]:
                return x, y
    return None


def _trace_boundary(mask, height: int, width: int) -> list[tuple[int, int]] | None:
    """Moore-neighbor tracing, keeping the last BACKGROUND cell examined —
    the clockwise scan around each new pixel starts from that background cell,
    which is what keeps the walk glued to the boundary."""
    start = _first_foreground(mask, height, width)
    if start is None:
        return None

    def on(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and bool(mask[y][x])

    boundary = [start]
    p = start
    bg = (start[0] - 1, start[1])  # west neighbor: background by scan order
    first_step: tuple[tuple[int, int], tuple[int, int]] | None = None
    limit = 4 * height * width

    for _ in range(limit):
        idx0 = _MOORE.index((bg[0] - p[0], bg[1] - p[1]))
        found = None
        for step in range(1, 9):
            idx = (idx0 + step) % 8
            nxt = (p[0] + _MOORE[idx][0], p[1] + _MOORE[idx][1])
            if on(*nxt):
                found = (idx, nxt)
                break
        if found is None:
            return boundary  # isolated single pixel
        idx, nxt = found
        # stop when we are about to repeat the very first move
        if first_step is None:
            first_step = (p, nxt)
        elif (p, nxt) == first_step:
            break
        prev_idx = (idx - 1) % 8
        bg = (p[0] + _MOORE[prev_idx][0], p[1] + _MOORE[prev_idx][1])
        p = nxt
        if p != boundary[-1]:
            boundary.append(p)
    if len(boundary) > 1 and boundary[-1] == boundary[0]:
        boundary.pop()
    return boundary


def _perpendicular_distance(pt, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = pt, a, b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    return abs(dy * px - dx * py + bx * ay - by * ax) / (dx * dx + dy * dy) ** 0.5


def _douglas_peucker(points: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    if len(points) < 3:
        return points
    stack = [(0, len(points) - 1)]
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    while stack:
        first, last = stack.pop()
        max_dist, index = 0.0, first
        for i in range(first + 1, last):
            dist = _perpendicular_distance(points[i], points[first], points[last])
            if dist > max_dist:
                max_dist, index = dist, i
        if max_dist > epsilon:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [p for p, k in zip(points, keep, strict=True) if k]


def mask_to_shape(
    mask, *, epsilon: float = 1.5, min_points: int = 3, anchor: tuple[int, int] | None = None
) -> MaskShape | None:
    """Trace the mask's largest blob — or the blob under `anchor` when that
    pixel is foreground — and return its simplified flat polygon
    [x1, y1, x2, y2, ...] with the blob's box and area, or None when the mask
    is empty or the blob is degenerate (fewer than `min_points` corners)."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    if not height or not width:
        return None
    found = largest_blob(mask, anchor)
    if found is None:
        return None
    blob, area, (x0, y0, x1, y1) = found
    boundary = _trace_boundary(_blob_mask(blob, height, width), height, width)
    if boundary is None or len(boundary) < min_points:
        return None
    # close the ring for DP, then drop the duplicate endpoint
    ring = [*boundary, boundary[0]]
    simplified = _douglas_peucker(ring, epsilon)[:-1]
    if len(simplified) < min_points:
        return None
    flat: list[float] = []
    for x, y in simplified:
        flat.extend((float(x), float(y)))
    return MaskShape(
        polygon=flat,
        bbox=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
        area=area,
    )


def simplify_polygon(flat: list[float], max_points: int) -> list[float]:
    """The same polygon with at most `max_points` vertices (never fewer than
    3): Douglas-Peucker on the closed ring with the smallest tolerance that
    gets under the cap, found by bisection. The ring is cut at the vertex
    farthest from the first one so both halves keep their end points and a
    huge tolerance still leaves a triangle, not a line. Annotators use this
    when a mask's outline is more detailed than the label needs (SAM-T4)."""
    max_points = max(3, int(max_points))
    pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
    if len(pts) <= max_points:
        return list(flat)
    far = max(range(1, len(pts)), key=lambda i: _perpendicular_distance(pts[i], pts[0], pts[0]))
    halves = (pts[: far + 1], [*pts[far:], pts[0]])

    def reduced(epsilon: float) -> list[tuple[float, float]]:
        a = _douglas_peucker(halves[0], epsilon)
        b = _douglas_peucker(halves[1], epsilon)
        return [*a[:-1], *b[:-1]]  # each half's last point opens the next half / the ring

    lo, hi = 0.0, max(abs(x) + abs(y) for x, y in pts) + 1.0
    best: list[tuple[float, float]] | None = None
    for _ in range(40):
        mid = (lo + hi) / 2
        candidate = reduced(mid)
        if len(candidate) > max_points:
            lo = mid  # still too detailed: coarser
        else:
            hi = mid  # fits (or collapsed): finer next, remember it if it is a polygon
            if len(candidate) >= 3:
                best = candidate
    if best is None:
        # no tolerance yields a polygon under the cap (a rectangle capped at 3
        # is the classic case): the triangle of the two cut points and the
        # vertex farthest from the line between them
        a, b = pts[0], pts[far]
        third = max((q for q in pts if q not in (a, b)),
                    key=lambda q: _perpendicular_distance(q, a, b), default=None)
        best = [a, b] if third is None else sorted(
            [a, b, third], key=lambda q: pts.index(q)
        )
    out: list[float] = []
    for x, y in best:
        out.extend((float(x), float(y)))
    return out


def mask_to_polygon(mask, *, epsilon: float = 1.5, min_points: int = 3) -> list[float] | None:
    """The polygon of `mask_to_shape`, or None."""
    shape = mask_to_shape(mask, epsilon=epsilon, min_points=min_points)
    return shape.polygon if shape is not None else None
