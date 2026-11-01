// Convert a stepper's planned motion into phase stepping segments
//
// The MCU updates the coil currents on a fixed grid of ticks and integrates
// queued constant acceleration segments in integer phase units (see
// src/phase_stepper.c). This code samples the stepper position (including
// kinematics and input shaping) at every tick and fits such segments to it,
// tracking the MCU's integer state exactly so rounding never accumulates.
//
// Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h> // fabs
#include <stddef.h> // offsetof
#include <stdlib.h> // malloc
#include <string.h> // memset
#include "compiler.h" // __visible
#include "itersolve.h" // struct stepper_kinematics
#include "phasegen.h" // struct phasegen_seg
#include "trapq.h" // struct move

// Stationary ticks needed before the queue is left empty (motor held)
#define IDLE_TICKS 20
#define VEL_LIMIT 2147483647.

struct phasegen {
    struct stepper_kinematics *sk;
    double tick_time, phase_per_mm, tolerance;
    // Tick n latches at tick n+1, so its target is the position then:
    // ref_phase + (position(grid_time + (n+1)*tick_time) - ref_pos) * ppm
    double grid_time, ref_pos;
    int64_t ref_phase;
    // The MCU state after last_tick: phase (unwrapped) and whether queued
    // segments are still moving the motor
    int64_t last_tick, phase;
    int moving;
    struct move *m;
    double *delta;
    int delta_size;
};

struct phasegen * __visible
phasegen_alloc(void)
{
    struct phasegen *pg = malloc(sizeof(*pg));
    memset(pg, 0, sizeof(*pg));
    return pg;
}

void __visible
phasegen_free(struct phasegen *pg)
{
    if (!pg)
        return;
    free(pg->delta);
    free(pg);
}

void __visible
phasegen_setup(struct phasegen *pg, struct stepper_kinematics *sk
               , double tick_time, double phase_per_mm, double tolerance)
{
    pg->sk = sk;
    pg->tick_time = tick_time;
    pg->phase_per_mm = phase_per_mm;
    pg->tolerance = tolerance;
}

// Start from the MCU state after start_tick, where the motor holds
// start_phase at stepper position ref_pos
void __visible
phasegen_start(struct phasegen *pg, double grid_time, int64_t start_tick
               , int64_t start_phase, double ref_pos)
{
    pg->grid_time = grid_time;
    pg->last_tick = start_tick;
    pg->phase = pg->ref_phase = start_phase;
    pg->ref_pos = ref_pos;
    pg->moving = 0;
}

double __visible
phasegen_target(struct phasegen *pg, int64_t tick)
{
    double t = pg->grid_time + (tick + 1) * pg->tick_time;
    struct move *m = pg->m;
    if (!m) {
        trapq_check_sentinels(pg->sk->tq);
        m = list_first_entry(&pg->sk->tq->moves, struct move, node);
    }
    while (t >= m->print_time + m->move_t)
        m = list_next_entry(m, node);
    while (t < m->print_time)
        m = list_prev_entry(m, node);
    pg->m = m;
    double pos = pg->sk->calc_position_cb(pg->sk, m, t - m->print_time);
    return pg->ref_phase + (pos - pg->ref_pos) * pg->phase_per_mm;
}

int64_t __visible
phasegen_get_phase(struct phasegen *pg)
{
    return pg->phase;
}

int64_t __visible
phasegen_get_last_tick(struct phasegen *pg)
{
    return pg->last_tick;
}

// Least squares integer fit of d[k] ~ k*vel + k*k*half for k = 1..count
static void
fit_segment(const double *d, int count, int64_t *pvel, int64_t *phalf)
{
    if (count == 1) {
        *phalf = 0;
        *pvel = llround(d[1]);
        return;
    }
    double s2 = 0., s3 = 0., s4 = 0., b1 = 0., b2 = 0.;
    int k;
    for (k = 1; k <= count; k++) {
        double kk = (double)k * k;
        s2 += kk;
        s3 += kk * k;
        s4 += kk * kk;
        b1 += k * d[k];
        b2 += kk * d[k];
    }
    // Round the acceleration first, then refit the velocity around it
    double half = (s2 * b2 - s3 * b1) / (s2 * s4 - s3 * s3);
    *phalf = llround(half);
    *pvel = llround((b1 - *phalf * s3) / s2);
}

static int
segment_ok(struct phasegen *pg, const double *d, int count, int64_t vel
           , int64_t half)
{
    double end_vel = vel + 2. * half * count;
    if (fabs(vel) > VEL_LIMIT || fabs(half) > VEL_LIMIT
        || fabs(end_vel) > VEL_LIMIT)
        return 0;
    int k;
    for (k = 1; k <= count; k++) {
        double kk = (double)k;
        if (fabs(kk * vel + kk * kk * half - d[k]) > pg->tolerance)
            return 0;
    }
    return 1;
}

// Find the longest segment from the current state that stays in tolerance
static int
longest_segment(struct phasegen *pg, const double *d, int avail
                , int64_t *pvel, int64_t *phalf)
{
    int good = 0, bad = avail + 1, count = 1;
    int64_t vel, half;
    for (;;) {
        fit_segment(d, count, &vel, &half);
        if (!segment_ok(pg, d, count, vel, half)) {
            bad = count;
            break;
        }
        good = count;
        *pvel = vel;
        *phalf = half;
        if (count == avail)
            break;
        count = count * 2 > avail ? avail : count * 2;
    }
    while (bad - good > 1) {
        count = (good + bad) / 2;
        fit_segment(d, count, &vel, &half);
        if (segment_ok(pg, d, count, vel, half)) {
            good = count;
            *pvel = vel;
            *phalf = half;
        } else {
            bad = count;
        }
    }
    return good;
}

static int
add_seg(struct phasegen_seg *out, int32_t *pnum, int32_t max_out
        , uint32_t count, int32_t vel, int32_t half, int64_t wait_tick)
{
    if (*pnum >= max_out)
        return -1;
    struct phasegen_seg *s = &out[(*pnum)++];
    s->count = count;
    s->start_vel = vel;
    s->half_accel = half;
    s->wait_tick = wait_tick;
    return 0;
}

// Emit segments for every tick whose latch time is at or before flush_time.
// Returns the number of entries written to out, or -1 if out is too small
// or a segment cannot be represented.
int32_t __visible
phasegen_generate(struct phasegen *pg, double flush_time
                  , struct phasegen_seg *out, int32_t max_out)
{
    int64_t end_tick = (int64_t)floor(
        (flush_time - pg->grid_time) / pg->tick_time) - 1;
    int64_t total = end_tick - pg->last_tick;
    if (total <= 0)
        return 0;
    if (total + 1 > pg->delta_size) {
        free(pg->delta);
        pg->delta_size = total + 1;
        pg->delta = malloc(sizeof(*pg->delta) * pg->delta_size);
    }
    pg->m = NULL;
    double *target = pg->delta;
    int64_t i;
    for (i = 1; i <= total; i++)
        target[i] = phasegen_target(pg, pg->last_tick + i);

    int32_t num = 0;
    int64_t done = 0;
    double tol = pg->tolerance;
    while (done < total) {
        // Count ticks ahead that stay at the current phase
        int64_t still = 0;
        while (done + still < total
               && fabs(target[done + still + 1] - pg->phase) <= tol)
            still++;
        if (!pg->moving) {
            if (still) {
                done += still;
                continue;
            }
            if (add_seg(out, &num, max_out, 0, 0, 0
                        , pg->last_tick + done + 1))
                return -1;
            pg->moving = 1;
        } else if (still >= IDLE_TICKS || done + still == total) {
            if (still) {
                // A one tick segment at zero velocity leaves the MCU at rest
                if (add_seg(out, &num, max_out, 1, 0, 0, 0))
                    return -1;
                pg->moving = 0;
                done++;
                continue;
            }
        }
        // Fit the next segment relative to the current MCU phase
        int64_t avail = total - done;
        double *d = &target[done];
        double base = pg->phase, saved = d[0];
        for (i = 1; i <= avail; i++)
            d[i] -= base;
        int64_t vel, half;
        int count = longest_segment(pg, d, avail, &vel, &half);
        for (i = 1; i <= avail; i++)
            d[i] += base;
        d[0] = saved;
        if (!count)
            return -1;
        if (add_seg(out, &num, max_out, count, vel, half, 0))
            return -1;
        pg->phase += count * vel + (int64_t)count * count * half;
        done += count;
    }
    pg->last_tick = end_tick;
    return num;
}
