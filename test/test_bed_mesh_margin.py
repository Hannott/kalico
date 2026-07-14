import pytest

from klippy.extras import bed_mesh


class _GCodeError(Exception):
    pass


def _err(msg):
    raise _GCodeError(msg)


class _FakeProbe:
    def __init__(self, x_offset, y_offset):
        self._offsets = (x_offset, y_offset, 0.0)

    def get_offsets(self):
        return self._offsets


class _FakePrinter:
    def __init__(self, probe=None):
        self._probe = probe

    def lookup_object(self, name, default=None):
        if name == "probe":
            return self._probe
        raise KeyError(name)


def _make_calibrate(mesh_margin, probe=None):
    bmc = bed_mesh.BedMeshCalibrate.__new__(bed_mesh.BedMeshCalibrate)
    bmc.mesh_margin = mesh_margin
    bmc.printer = _FakePrinter(probe)
    bmc.radius = None
    return bmc


class TestMarginBounds:
    """Pure math: (axis_min, axis_max, offset) -> inset bounds for one axis.

    A mesh point P is reachable only if the toolhead can move to P - offset,
    so P must simultaneously satisfy:
      - the margin standoff:  axis_min + margin <= P <= axis_max - margin
      - reachability:         axis_min + offset <= P <= axis_max + offset
    _margin_bounds intersects these two constraints (the more restrictive
    bound wins on each side; they are never additive).
    """

    def test_no_offset_no_margin_is_noop(self):
        bmc = _make_calibrate(mesh_margin=0.0)
        assert bmc._margin_bounds(_err, 0.0, 235.0, 0.0) == (0.0, 235.0)

    def test_no_offset_applies_symmetric_margin(self):
        bmc = _make_calibrate(mesh_margin=25.0)
        assert bmc._margin_bounds(_err, 0.0, 235.0, 0.0) == (25.0, 210.0)

    def test_margin_zero_reduces_to_pure_reachability(self):
        # Must match Part-A's safe-bounds formula exactly when margin=0.
        bmc = _make_calibrate(mesh_margin=0.0)
        assert bmc._margin_bounds(_err, 0.0, 235.0, 25.0) == (25.0, 235.0)
        assert bmc._margin_bounds(_err, 0.0, 235.0, -25.0) == (0.0, 210.0)

    def test_margin_equal_to_offset(self):
        # The exact scenario from the design discussion: offset == margin.
        # The near side is pinned by reachability alone (toolhead ends up
        # exactly at axis_min); the far side still has to pull in, because
        # reaching the full margin standoff there requires the toolhead at
        # axis_max - offset - margin, i.e. axis_max - 50.
        bmc = _make_calibrate(mesh_margin=25.0)
        mesh_min, mesh_max = bmc._margin_bounds(_err, 0.0, 235.0, 25.0)
        assert (mesh_min, mesh_max) == (25.0, 210.0)
        assert mesh_min - 25.0 == 0.0  # toolhead sits at axis_min
        assert mesh_max - 25.0 == 185.0  # toolhead sits at axis_max - 50

    def test_margin_larger_than_offset(self):
        bmc = _make_calibrate(mesh_margin=40.0)
        assert bmc._margin_bounds(_err, 0.0, 235.0, 25.0) == (40.0, 195.0)

    def test_margin_smaller_than_offset(self):
        bmc = _make_calibrate(mesh_margin=10.0)
        assert bmc._margin_bounds(_err, 0.0, 235.0, 25.0) == (25.0, 225.0)

    def test_negative_offset_flips_which_side_is_pinned(self):
        bmc = _make_calibrate(mesh_margin=25.0)
        pos = bmc._margin_bounds(_err, 0.0, 235.0, 25.0)
        neg = bmc._margin_bounds(_err, 0.0, 235.0, -25.0)
        # Same mesh bounds either way (margin == abs(offset) here), but a
        # different side ends up pinned exactly at the axis limit.
        assert pos == neg == (25.0, 210.0)
        assert pos[0] - 25.0 == 0.0  # +offset pins the min side
        assert neg[1] - (-25.0) == 235.0  # -offset pins the max side

    def test_raises_when_margin_and_offset_leave_no_room(self):
        bmc = _make_calibrate(mesh_margin=150.0)
        with pytest.raises(_GCodeError, match="mesh_margin"):
            bmc._margin_bounds(_err, 0.0, 235.0, 25.0)


class TestApplyMargin:
    """Wiring: probe lookup + METHOD handling + mesh_min/mesh_max tuples."""

    def test_applies_probe_offsets_per_axis(self):
        # x: offset(25) == margin(25) -> far side pinned by margin alone
        #    (reachability and margin agree exactly).
        # y: offset(-40) exceeds margin(25) in magnitude -> reachability
        #    dominates the near side, forcing it in further than margin
        #    alone would.
        probe = _FakeProbe(x_offset=25.0, y_offset=-40.0)
        bmc = _make_calibrate(mesh_margin=25.0, probe=probe)
        bmc.mesh_min = (0.0, 0.0)
        bmc.mesh_max = (235.0, 235.0)
        bmc._apply_margin(_err)
        assert bmc.mesh_min == (25.0, 25.0)
        assert bmc.mesh_max == (210.0, 195.0)

    def test_manual_method_ignores_probe_offsets(self):
        probe = _FakeProbe(x_offset=25.0, y_offset=25.0)
        bmc = _make_calibrate(mesh_margin=25.0, probe=probe)
        bmc.mesh_min = (0.0, 0.0)
        bmc.mesh_max = (235.0, 235.0)
        bmc._apply_margin(_err, probe_method="manual")
        # No probe offset applied -> plain symmetric margin.
        assert bmc.mesh_min == (25.0, 25.0)
        assert bmc.mesh_max == (210.0, 210.0)

    def test_no_probe_object_behaves_like_zero_offset(self):
        bmc = _make_calibrate(mesh_margin=25.0, probe=None)
        bmc.mesh_min = (0.0, 0.0)
        bmc.mesh_max = (235.0, 235.0)
        bmc._apply_margin(_err)
        assert bmc.mesh_min == (25.0, 25.0)
        assert bmc.mesh_max == (210.0, 210.0)


class TestMarginKeepsPointsReachable:
    """Property check: for arbitrary axis/offset/margin combos, every bound
    _margin_bounds produces (a) never asks the toolhead to leave the axis
    range, and (b) matches an independently-derived reference formula based
    on Part-A's per-side reachability bound.
    """

    @pytest.mark.parametrize(
        "axis_min,axis_max,offset,margin",
        [
            (0.0, 235.0, 25.0, 25.0),
            (0.0, 235.0, -25.0, 25.0),
            (0.0, 350.0, 5.0, 2.0),
            (-10.0, 300.0, -8.0, 15.0),
            (0.0, 235.0, 0.0, 0.0),
            (0.0, 235.0, 25.0, 0.0),
            (0.0, 235.0, 0.0, 25.0),
        ],
    )
    def test_reachable_and_matches_reference_formula(
        self, axis_min, axis_max, offset, margin
    ):
        bmc = _make_calibrate(mesh_margin=margin)
        mesh_min, mesh_max = bmc._margin_bounds(
            _err, axis_min, axis_max, offset
        )
        for point in (mesh_min, mesh_max):
            carriage = point - offset
            assert axis_min - 1e-9 <= carriage <= axis_max + 1e-9

        # Reference formula, derived independently from Part-A's per-side
        # safe bound (safe_min = axis_min + max(offset, 0), safe_max =
        # axis_max + min(offset, 0)), intersected with the plain symmetric
        # margin bound.
        reach_min = axis_min + max(offset, 0.0)
        reach_max = axis_max + min(offset, 0.0)
        expected_min = max(axis_min + margin, reach_min)
        expected_max = min(axis_max - margin, reach_max)
        assert mesh_min == pytest.approx(expected_min)
        assert mesh_max == pytest.approx(expected_max)
