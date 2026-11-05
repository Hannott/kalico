# Tests for resonance_ramp.py: the resonance-weighted notch-pair search and
# the jerk-limited ramp construction it drives.
#
# Run: klippy-env/bin/python test/test_resonance_ramp.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "klippy", "extras"))
import resonance_ramp as rr  # noqa: E402


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


def test_saturated_rise_freq_keeps_the_quieter_zero():
    f_lo, f_hi, a = 55.0, 75.0, 8000.0
    seen = set()
    for dv in (120.0, 150.0, 200.0, 300.0, 400.0, 600.0):
        f_rise = rr.saturated_rise_freq(dv, f_lo, f_hi, a)
        assert f_rise in (f_lo, f_hi)
        plateau = dv / a
        other = f_lo if f_rise == f_hi else f_hi
        kept = abs(rr._sinc(other / f_rise) * rr._sinc(other * plateau))
        dropped = abs(rr._sinc(f_rise / other) * rr._sinc(f_rise * plateau))
        assert kept <= dropped + 1e-12, (dv, f_rise, kept, dropped)
        seen.add(f_rise)
    assert seen == {f_lo, f_hi}
    assert rr.saturated_rise_freq(300.0, 55.0, 55.0, a) == 55.0
    print("  saturation keeps whichever zero leaves less residual OK")


def test_saturated_jerk_and_duration_follow_the_chosen_zero():
    f_lo, f_hi, a, dv = 55.0, 75.0, 8000.0, 300.0
    f_rise = rr.saturated_rise_freq(dv, f_lo, f_hi, a)
    assert abs(rr.ramp_jerk(dv, f_lo, f_hi, a) - a * f_rise) < 1e-9
    want = dv / a + 1.0 / f_rise
    assert abs(rr.notch_duration(dv, f_lo, f_hi, a) - want) < 1e-12
    # The fixed-step integrator lands short of the closed form by about
    # one step per edge, so check at a fine step.
    segs, ok = rr._ramp_up(0.0, dv, f_lo, f_hi, a, 0.0001, 4096)
    assert ok
    assert abs(sum(s[0] for s in segs) - want) < 0.0005
    assert abs(max(s[5] for s in segs) - a) < 1e-9
    print("  saturated jerk and duration follow the chosen zero OK")


def test_slide_notch_pair_finds_the_smallest_factor_that_fits():
    vs, vc, ve, f_lo, f_hi, a = 0.0, 100.0, 0.0, 55.0, 75.0, 20000.0
    need = rr.profile_dist(vs, vc, ve, f_lo, f_hi, a)
    assert rr.slide_notch_pair(vs, vc, ve, 1.01 * need, f_lo, f_hi, a, 2.0) == (
        (f_lo, f_hi),
        1.0,
    )
    move_d = 0.8 * need
    (s_lo, s_hi), k = rr.slide_notch_pair(
        vs, vc, ve, move_d, f_lo, f_hi, a, 2.0
    )
    assert 1.0 < k <= 2.0
    assert abs(s_lo / f_lo - k) < 1e-9 and abs(s_hi / f_hi - k) < 1e-9
    assert rr.profile_dist(vs, vc, ve, s_lo, s_hi, a) <= move_d
    k_tight = 0.999 * k
    assert rr.profile_dist(vs, vc, ve, f_lo * k_tight, f_hi * k_tight, a) > (
        move_d * rr.SLIDE_FIT_MARGIN
    )
    segs = rr.build_notch_profile(vs, vc, ve, move_d, s_lo, s_hi, a, 0.0002)
    check_segs(segs, move_d, vs, ve, "slid")
    print("  the slide picks the smallest factor that fits the move OK")


def test_slide_notch_pair_is_bounded():
    vs, vc, ve, f_lo, f_hi, a = 0.0, 100.0, 0.0, 55.0, 75.0, 20000.0
    need = rr.profile_dist(vs, vc, ve, f_lo, f_hi, a)
    assert (
        rr.slide_notch_pair(vs, vc, ve, 0.3 * need, f_lo, f_hi, a, 2.0) is None
    )
    assert (
        rr.slide_notch_pair(vs, vc, ve, 0.8 * need, f_lo, f_hi, a, 1.0) is None
    )
    print("  the slide refuses past max_slide and when disabled OK")


def test_notch_reach_is_tight_and_the_emitter_can_render_it():
    pairs = ((55.0, 55.0), (45.2, 76.4), (55.0, 75.0), (30.0, 120.0))
    checked = 0
    for f_lo, f_hi in pairs:
        for a in (3000.0, 8000.0, 20000.0):
            for v1 in (0.0, 5.0, 40.0, 150.0):
                for d in (0.2, 1.0, 5.0, 30.0, 200.0):
                    v0 = rr.notch_reach(v1, d, f_lo, f_hi, a)
                    assert v0 >= v1
                    need = rr.notch_dist(v0, v1, f_lo, f_hi, a)
                    assert need <= d * (1.0 + 1e-6), (f_lo, f_hi, a, v1, d)
                    if v0 > v1 + 1e-6:
                        # Tight, except where the saturated rise edge
                        # switches zeros just above v0 and the ramp
                        # distance drops discontinuously.
                        v_more = v0 * 1.001 + 0.01
                        over = rr.notch_dist(v_more, v1, f_lo, f_hi, a)
                        switched = rr.ramp_jerk(
                            v_more - v1, f_lo, f_hi, a
                        ) != rr.ramp_jerk(v0 - v1, f_lo, f_hi, a)
                        assert over > d or switched, (f_lo, f_hi, a, v1, d)
                        segs = rr.build_notch_profile(
                            v0, v0, v1, d, f_lo, f_hi, a, 0.001
                        )
                        assert segs is not None, (f_lo, f_hi, a, v1, d, v0)
                        check_segs(segs, d, v0, v1, "reach")
                    checked += 1
    assert checked == 240
    print("  notch_reach is tight and always renderable OK")


def test_notch_peak_is_tight_and_the_emitter_can_render_it():
    f_lo, f_hi = 45.2, 76.4
    for a in (3000.0, 20000.0):
        for vs, ve in ((0.0, 0.0), (20.0, 60.0), (100.0, 10.0), (80.0, 80.0)):
            for d in (2.0, 10.0, 60.0, 400.0):
                need = rr.notch_dist(vs, ve, f_lo, f_hi, a)
                if need > d:
                    continue
                vc = rr.notch_peak(vs, ve, d, f_lo, f_hi, a)
                assert vc >= max(vs, ve)
                used = rr.profile_dist(vs, vc, ve, f_lo, f_hi, a)
                assert used <= d * (1.0 + 1e-6)
                if vc > max(vs, ve) + 1e-6:
                    over = rr.profile_dist(
                        vs, vc * 1.001 + 0.01, ve, f_lo, f_hi, a
                    )
                    assert over > d
                segs = rr.build_notch_profile(
                    vs, vc, ve, d, f_lo, f_hi, a, 0.001
                )
                assert segs is not None, (a, vs, ve, d, vc)
                check_segs(segs, d, vs, ve, "peak")
    print("  notch_peak is tight and always renderable OK")


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
    test_saturated_rise_freq_keeps_the_quieter_zero()
    test_saturated_jerk_and_duration_follow_the_chosen_zero()
    test_slide_notch_pair_finds_the_smallest_factor_that_fits()
    test_slide_notch_pair_is_bounded()
    test_notch_reach_is_tight_and_the_emitter_can_render_it()
    test_notch_peak_is_tight_and_the_emitter_can_render_it()
    test_finer_jerk_dt_still_lands_exactly()
    print("ALL PASS")


if __name__ == "__main__":
    main()
