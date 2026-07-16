#ifndef __KIN_SHAPER_H
#define __KIN_SHAPER_H

// Largest single shaper (3hump_ei) has 5 impulses. A multimode shaper
// convolves multiple shapers together (impulse counts multiply), so the
// worst case for 2 modes is 5 x 5 = 25; keep some headroom above that.
#define MAX_SHAPER_PULSES 32

struct shaper_pulses {
    int num_pulses;
    struct {
        double t, a;
    } pulses[MAX_SHAPER_PULSES];
};

struct move;

int init_shaper(int n, double a[], double t[], struct shaper_pulses *sp);
double shaper_calc_position(const struct move *m, int axis, double move_time
                            , const struct shaper_pulses *sp);

#endif  // kin_shaper.h
