# Tests for resonance_ramp.py: the resonance-weighted notch-pair search and
# the jerk-limited ramp construction it drives.
#
# Run: klippy-env/bin/python test/test_resonance_ramp.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "klippy", "extras"))
import resonance_ramp as rr


def test_peak_response_peaks_at_f0():
    model = rr.PeakModel([(55.0, 0.05, 1.0)])
    at_peak = model.weight(55.0)
    for off in (10.0, 20.0, 40.0):
        assert model.weight(55.0 - off) < at_peak
        assert model.weight(55.0 + off) < at_peak
    print("  peak response is maximal at its own center frequency OK")


def test_peak_model_discards_invalid_entries():
    model = rr.PeakModel(
        [(55.0, 0.05, 1.0), (0.0, 0.05, 1.0), (60.0, 0.0, 0.0)]
    )
    assert len(model.peaks) == 1
    assert model.dominant_freq() == 55.0
    print("  invalid/zero-weight peaks are discarded OK")


def test_dominant_freq_is_the_highest_weight_peak():
    model = rr.PeakModel(
        [(55.0, 0.05, 0.4), (90.0, 0.05, 1.0), (30.0, 0.05, 0.2)]
    )
    assert model.dominant_freq() == 90.0
    print("  dominant_freq picks the highest-weight peak OK")


def test_single_peak_search_returns_its_own_frequency():
    model = rr.PeakModel([(55.0, 0.05, 1.0)])
    pair = rr.best_notch_pair(model)
    assert pair == (55.0, 55.0)
    print("  a single-peak model returns that peak as a single-zero pair OK")


def test_two_peak_search_beats_either_peak_alone():
    model = rr.PeakModel([(55.0, 0.03, 1.0), (90.0, 0.03, 0.8)])
    pair = rr.best_notch_pair(model)
    best_score = rr.score_pair(pair[0], pair[1], model)
    for f in (55.0, 90.0):
        assert best_score <= rr.score_pair(f, f, model) + 1e-12
    print("  the searched pair scores at least as well as either peak alone OK")


def test_a_second_real_peak_is_not_outvoted_by_the_tallest_peaks_skirt():
    # Regression: scoring used to sum |A(f)|^2 over one shared frequency
    # grid, which let the tallest peak's wide, high-response skirt dominate.
    # On this real 3-peak machine that picked a DOUBLE zero on 45.2 Hz,
    # leaving 2.4% at the nearly-as-tall 76.4 Hz peak and costing a 44 ms
    # ramp -- when nulling 45.2 AND 76.4 exactly is both quieter at every
    # measured peak and a shorter (35 ms) ramp. Scoring now averages per
    # peak, so each measured mode is judged on its own band.
    model = rr.PeakModel(
        [(45.2, 0.055, 1.00), (76.4, 0.050, 0.89), (137.2, 0.080, 0.38)]
    )
    f_lo, f_hi = rr.best_notch_pair(model)
    assert (f_lo, f_hi) == (45.2, 76.4), (f_lo, f_hi)
    # Both of the two dominant modes are actually nulled, not just the top.
    for f in (45.2, 76.4):
        assert rr.ramp_spectrum(f, f_lo, f_hi) < 1e-9, f
    # ...and it is not paying for that with a longer ramp than the
    # double-zero-on-the-tallest-peak shape it replaced.
    assert rr.notch_duration(100.0, f_lo, f_hi) < rr.notch_duration(
        100.0, 45.2, 45.2
    )
    print(
        "  a genuine second peak is not outvoted by the tallest one's skirt OK"
    )


def test_search_ignores_frequencies_below_the_floor():
    model = rr.PeakModel([(3.0, 0.05, 1.0), (60.0, 0.05, 0.5)])
    pair = rr.best_notch_pair(model, min_freq=5.0)
    assert 3.0 not in pair
    print("  peaks below min_freq are excluded from the search OK")


def test_search_returns_none_for_an_empty_model():
    model = rr.PeakModel([])
    assert rr.best_notch_pair(model) is None
    print("  an empty peak model has nothing to search OK")


def test_ramp_spectrum_null_at_configured_frequencies():
    # |sinc(f/f_hi)*sinc(f/f_lo)| is exactly zero at f_lo and f_hi (nonzero
    # integer argument to sinc), for both the single- and dual-zero cases.
    assert rr.ramp_spectrum(55.0, 55.0, 55.0) < 1e-9
    assert rr.ramp_spectrum(55.0, 55.0, 75.0) < 1e-9
    assert rr.ramp_spectrum(75.0, 55.0, 75.0) < 1e-9
    print("  the ideal ramp spectrum has exact zeros at f_lo and f_hi OK")


def test_single_zero_duration_and_peak_accel():
    f = 55.0
    dv = 100.0
    assert abs(rr.notch_duration(dv, f, f) - 2.0 / f) < 1e-12
    j = rr.ramp_jerk(dv, f, f)
    assert abs(j - dv * f * f) < 1e-9
    print("  single-zero duration/jerk match the closed form OK")


def test_dual_zero_collapses_to_single_when_equal():
    assert rr.ramp_jerk(100.0, 55.0, 55.0) == rr.ramp_jerk(100.0, 55.0, 55.00)
    d1 = rr.notch_duration(100.0, 55.0, 75.0)
    d2 = 1.0 / 55.0 + 1.0 / 75.0
    assert abs(d1 - d2) < 1e-12
    print("  dual-zero duration matches 1/f_lo + 1/f_hi OK")


def check_segs(segs, move_d, want_vs, want_ve, tag):
    # Slice-tuple convention (matches trapq_append's own accel_t/cruise_t/
    # decel_t phases): an ACCEL slice (at>0) runs sv -> cv over `at`; a
    # CRUISE slice (ct>0) holds sv==cv==cruise speed over `ct`; a DECEL
    # slice (dt>0) stores sv==cv==the speed AT THE START of deceleration,
    # and actually runs from cv down to cv - a*dt over `dt`.
    assert segs, tag
    v = want_vs
    total_d = 0.0
    for at, ct, dt, sv, cv, a, dist in segs:
        assert at >= -1e-9 and ct >= -1e-9 and dt >= -1e-9, tag
        assert sv >= -1e-9 and cv >= -1e-9, tag
        assert abs(sv - v) < 1e-6, (tag, sv, v)
        if at > 0.0:
            implied = 0.5 * (sv + cv) * at
            v_end = cv
        elif ct > 0.0:
            implied = cv * ct
            v_end = cv
        elif dt > 0.0:
            v_after = cv - a * dt
            implied = 0.5 * (cv + v_after) * dt
            v_end = v_after
        else:
            implied = 0.0
            v_end = v
        assert abs(implied - dist) < 1e-6 * max(1.0, dist), (tag, implied, dist)
        v = v_end
        total_d += dist
    assert abs(v - want_ve) < 1e-6, (tag, v, want_ve)
    assert abs(total_d - move_d) < 1e-6 * max(1.0, move_d), (
        tag,
        total_d,
        move_d,
    )


def test_build_notch_profile_basic_invariants():
    for vs, vc, ve, move_d in (
        (0.0, 100.0, 0.0, 50.0),
        (20.0, 150.0, 20.0, 80.0),
        (0.0, 60.0, 30.0, 15.0),
    ):
        segs = rr.build_notch_profile(
            vs, vc, ve, move_d, 55.0, 75.0, 20000.0, 0.001
        )
        check_segs(segs, move_d, vs, ve, (vs, vc, ve, move_d))
    print("  emitted profiles satisfy the stepguard invariants OK")


def test_build_notch_profile_respects_max_accel():
    segs = rr.build_notch_profile(
        0.0, 400.0, 0.0, 200.0, 55.0, 75.0, 3000.0, 0.001
    )
    assert segs is not None
    for at, _ct, dt, _sv, _cv, a, _dist in segs:
        assert a <= 3000.0 + 1e-6
    print("  saturated profiles never exceed max_accel OK")


def test_build_notch_profile_infeasible_move_returns_none():
    # A move far too short for the ramp's own minimum runway.
    segs = rr.build_notch_profile(
        0.0, 300.0, 0.0, 0.5, 55.0, 75.0, 20000.0, 0.001
    )
    assert segs is None
    print("  a move too short for the ramp returns None (caller falls back) OK")


def test_build_notch_profile_no_speed_change_is_pure_cruise():
    segs = rr.build_notch_profile(
        50.0, 50.0, 50.0, 20.0, 55.0, 75.0, 20000.0, 0.001
    )
    assert len(segs) == 1
    at, ct, dt, _sv, _cv, a, _dist = segs[0]
    assert at == 0.0 and dt == 0.0 and a == 0.0
    assert abs(ct - 20.0 / 50.0) < 1e-9
    print("  a move with no speed change emits a single cruise segment OK")


def test_finer_jerk_dt_still_lands_exactly():
    for dt in (0.001, 0.0005, 0.0002):
        segs = rr.build_notch_profile(
            0.0, 200.0, 0.0, 40.0, 55.0, 75.0, 20000.0, dt
        )
        check_segs(segs, 40.0, 0.0, 0.0, dt)
    print("  finer jerk_dt still lands exactly on distance/velocity OK")


def main():
    test_peak_response_peaks_at_f0()
    test_peak_model_discards_invalid_entries()
    test_dominant_freq_is_the_highest_weight_peak()
    test_single_peak_search_returns_its_own_frequency()
    test_two_peak_search_beats_either_peak_alone()
    test_a_second_real_peak_is_not_outvoted_by_the_tallest_peaks_skirt()
    test_search_ignores_frequencies_below_the_floor()
    test_search_returns_none_for_an_empty_model()
    test_ramp_spectrum_null_at_configured_frequencies()
    test_single_zero_duration_and_peak_accel()
    test_dual_zero_collapses_to_single_when_equal()
    test_build_notch_profile_basic_invariants()
    test_build_notch_profile_respects_max_accel()
    test_build_notch_profile_infeasible_move_returns_none()
    test_build_notch_profile_no_speed_change_is_pure_cruise()
    test_finer_jerk_dt_still_lands_exactly()
    print("ALL PASS")


if __name__ == "__main__":
    main()
