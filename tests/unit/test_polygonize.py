"""Mask -> polygon tracing/simplification (dependency-free, no ML needed)."""

from horos.backends.sam.polygonize import mask_to_polygon, simplify_polygon


def _grid(text: str) -> list[list[int]]:
    rows = [line.strip() for line in text.strip().splitlines()]
    return [[1 if ch == "#" else 0 for ch in row] for row in rows]


def _points(flat):
    return {(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)}


def test_solid_square():
    mask = _grid("""
    ........
    .#####..
    .#####..
    .#####..
    .#####..
    ........
    """)
    poly = mask_to_polygon(mask, epsilon=0.9)
    assert poly is not None
    corners = _points(poly)
    assert {(1, 1), (5, 1), (5, 4), (1, 4)} <= corners
    assert len(poly) // 2 <= 8  # simplification collapses the edges


def test_l_shape_keeps_concave_corner():
    mask = _grid("""
    #####....
    #####....
    #########
    #########
    """)
    poly = mask_to_polygon(mask, epsilon=0.9)
    assert poly is not None
    points = _points(poly)
    # not collapsed to the outer rectangle: the concave step around x=4..5 survives
    assert len(points) >= 6
    assert any(0 < x < 8 and y < 2 for x, y in points)
    assert (8, 2) in points or (8, 3) in points


def test_empty_mask_is_none():
    assert mask_to_polygon(_grid("....\n....")) is None


def test_single_pixel_is_none():
    assert mask_to_polygon(_grid("....\n.#..\n....")) is None


def test_polygon_is_flat_floats():
    poly = mask_to_polygon(_grid("###\n###\n###"), epsilon=0.5)
    assert poly is not None
    assert len(poly) % 2 == 0
    assert all(isinstance(v, float) for v in poly)


def test_works_with_numpy_masks():
    numpy = __import__("numpy")
    mask = numpy.zeros((10, 12), dtype=bool)
    mask[2:8, 3:10] = True
    poly = mask_to_polygon(mask, epsilon=0.9)
    assert poly is not None
    xs, ys = poly[0::2], poly[1::2]
    assert min(xs) == 3 and max(xs) == 9 and min(ys) == 2 and max(ys) == 7


# --- the largest blob wins (stray specks neither become the polygon nor widen the box)

def test_stray_speck_above_the_object_is_ignored():
    # a 1-pixel speck sits top-left, ABOVE the real blob: the old tracer picked
    # it (top-most foreground pixel) and returned a 3-corner sliver
    mask = _grid("""
    #.........
    ..........
    ...#####..
    ...#####..
    ...#####..
    ..........
    """)
    poly = mask_to_polygon(mask, epsilon=0.9)
    assert poly is not None
    xs, ys = poly[0::2], poly[1::2]
    assert min(xs) == 3 and max(xs) == 7 and min(ys) == 2 and max(ys) == 4


def test_shape_box_and_area_come_from_the_same_blob():
    from horos.backends.sam.polygonize import mask_to_shape

    mask = _grid("""
    ##........
    ##........
    ..........
    ...#####..
    ...#####..
    ...#####..
    .........#
    """)
    shape = mask_to_shape(mask, epsilon=0.9)
    assert shape is not None
    assert shape.bbox == (3.0, 3.0, 5.0, 3.0)  # the 5x3 blob, not the union of all three
    assert shape.area == 15
    xs, ys = shape.polygon[0::2], shape.polygon[1::2]
    assert (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1) == shape.bbox


def test_diagonally_touching_pixels_are_one_blob():
    from horos.backends.sam.polygonize import largest_blob

    mask = _grid("""
    ##...
    ##...
    ..#..
    ...##
    ...##
    """)
    found = largest_blob(mask)
    assert found is not None
    runs, area, box = found
    assert area == 9 and box == (0, 0, 5, 5)  # one 8-connected component


def test_largest_blob_on_numpy_masks_matches_lists():
    numpy = __import__("numpy")
    from horos.backends.sam.polygonize import largest_blob

    mask = numpy.zeros((20, 30), dtype=bool)
    mask[1, 1] = True            # speck
    mask[5:15, 10:25] = True     # object
    mask[18, 28:30] = True       # speck
    _, area, box = largest_blob(mask)
    assert area == 150 and box == (10, 5, 25, 15)
    assert largest_blob(mask.tolist()) == largest_blob(mask)


def test_only_specks_still_yields_the_biggest_one():
    from horos.backends.sam.polygonize import largest_blob

    _, area, box = largest_blob(_grid("#....\n...##\n....."))
    assert area == 2 and box == (3, 1, 5, 2)


def _ring(n, r=100.0):
    import math

    flat = []
    for k in range(n):
        a = 2 * math.pi * k / n
        flat.extend((200 + r * math.cos(a), 200 + r * math.sin(a)))
    return flat


def test_simplify_polygon_caps_the_vertex_count_with_a_subset_of_the_outline():
    ring = _ring(64)
    for cap in (48, 16, 8, 5, 3):
        out = simplify_polygon(ring, cap)
        assert 3 <= len(out) // 2 <= cap, cap
        pts = {(ring[i], ring[i + 1]) for i in range(0, len(ring), 2)}
        assert all((out[i], out[i + 1]) in pts for i in range(0, len(out), 2))
    # close to the cap, not far below it: the smallest tolerance that fits
    assert len(simplify_polygon(ring, 16)) // 2 >= 12


def test_simplify_polygon_leaves_small_polygons_alone_and_never_drops_below_three():
    square = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
    assert simplify_polygon(square, 4) == square
    assert simplify_polygon(square, 100) == square
    assert len(simplify_polygon(square, 3)) == 6
    assert len(simplify_polygon(square, 1)) == 6  # a cap below 3 means 3


def test_anchor_picks_the_blob_under_the_click_not_the_largest():
    """Two pieces of one object in a mask: without an anchor the bigger piece
    wins; with the click on the small piece, the small piece is the answer
    (the annotator building an object part by part must get the part they
    clicked, not the one they already kept)."""
    from horos.backends.sam.polygonize import mask_to_shape

    mask = [[0] * 20 for _ in range(10)]
    for y in range(1, 9):
        for x in range(1, 10):
            mask[y][x] = 1  # big piece, left
    for y in range(3, 6):
        for x in range(14, 18):
            mask[y][x] = 1  # small piece, right
    big = mask_to_shape(mask)
    assert big.bbox[0] == 1 and big.area == 72
    small = mask_to_shape(mask, anchor=(15, 4))
    assert small.bbox == (14.0, 3.0, 4.0, 3.0) and small.area == 12
    # an anchor on background falls back to the largest blob
    assert mask_to_shape(mask, anchor=(12, 4)).area == 72
