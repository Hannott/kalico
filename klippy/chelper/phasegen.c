// Convert a stepper's planned motion into phase stepping segments
//
// The MCU updates the coil currents on a fixed grid of ticks and integrates
// queued constant acceleration segments in integer phase units (see
// src/phase_stepper.c). This code samples the stepper position (including
// kinematics and input shaping) at every tick where it may move and fits
// such segments to it, tracking the MCU's integer state exactly so rounding
// never accumulates.
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
// Blocks of adaptive current peaks kept while a call works through them
// (a power of two above lead + lag + 1)
#define PEAK_CACHE 16

struct phasegen {
    struct stepper_kinematics *sk;
    double tick_time, phase_per_mm, tolerance;
    // Tick n latches at tick n+1, so its target is the position then:
    // ref_phase + (position(grid_time + (n+1)*tick_time) - ref_pos) * ppm
    double grid_time, ref_pos;
    int64_t ref_phase;
    // Position while no planned move covers the time (see stepper_position)
    double hold_pos;
    // The MCU state after last_tick: phase (unwrapped) and whether queued
    // segments are still moving the motor
    int64_t last_tick, phase;
    int moving;
    struct move *m;
    double *delta;
    int delta_size;
    // Adaptive current: a driver current scaler for each block of
    // block_ticks ticks, from the largest stepper acceleration within the
    // block and lead_blocks after and lag_blocks before it, sampled every
    // sample_ticks. Current is current_min + current_slope * |accel| within
    // current_min..current_max, and scaler 256 is full_scale.
    int32_t block_ticks, lead_blocks, lag_blocks, sample_ticks;
    double current_min, current_slope, current_max, full_scale;
    int32_t scaler_min, scaler_max, last_scaler;
    int64_t next_block;
    int64_t peak_block[PEAK_CACHE];
    double peak_accel[PEAK_CACHE];
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

// Stepper position at print time t, from the planned moves.
//
// Unlike step generation, phase stepping samples idle time too. Finished
// moves leave the trapq's move list and its head sentinel carries no
// position, so before the first listed move the motor is at rest where it
// last was. The same holds early in the stationary filler move trapq puts
// ahead of it (a second long, much more than any input shaper window),
// where shaping would blend in the head sentinel; later in the filler move
// shaping may already start the next motion.
static double
stepper_position(struct phasegen *pg, double t)
{
    struct move *m = pg->m;
    struct move *head = list_first_entry(&pg->sk->tq->moves, struct move
                                         , node);
    if (!m) {
        trapq_check_sentinels(pg->sk->tq);
        m = head;
    }
    while (t >= m->print_time + m->move_t)
        m = list_next_entry(m, node);
    while (t < m->print_time)
        m = list_prev_entry(m, node);
    pg->m = m;
    if (m == head || (list_prev_entry(m, node) == head && !m->start_v
                      && !m->half_accel
                      && t < m->print_time + 0.5 * m->move_t))
        return pg->hold_pos;
    pg->hold_pos = pg->sk->calc_position_cb(pg->sk, m, t - m->print_time);
    return pg->hold_pos;
}

// Start from the MCU state after start_tick, where the motor rests at
// stepper position ref_pos and holds start_phase
void __visible
phasegen_start(struct phasegen *pg, double grid_time, int64_t start_tick
               , int64_t start_phase, double ref_pos)
{
    pg->grid_time = grid_time;
    pg->last_tick = start_tick;
    pg->phase = pg->ref_phase = start_phase;
    pg->moving = 0;
    pg->m = NULL;
    pg->ref_pos = pg->hold_pos = ref_pos;
}

// The stepper position was redefined without motion (for example by
// SET_KINEMATIC_POSITION): keep the current phase at the new ref_pos
void __visible
phasegen_rebase(struct phasegen *pg, double ref_pos)
{
    pg->m = NULL;
    pg->ref_phase = pg->phase;
    pg->ref_pos = pg->hold_pos = ref_pos;
}

double __visible
phasegen_target(struct phasegen *pg, int64_t tick)
{
    double t = pg->grid_time + (tick + 1) * pg->tick_time;
    double pos = stepper_position(pg, t);
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

static int
drives_stepper(struct stepper_kinematics *sk, struct move *m)
{
    int af = sk->active_flags;
    return ((af & AF_X && m->axes_r.x != 0.)
            || (af & AF_Y && m->axes_r.y != 0.)
            || (af & AF_Z && m->axes_r.z != 0.));
}

// Earliest time from t on at which the stepper may leave its rest position,
// or end if it rests until then. Input shaping spreads a move over
// gen_steps_pre_active before and gen_steps_post_active after it.
static double
next_motion_time(struct stepper_kinematics *sk, double t, double end)
{
    if (!sk->tq)
        return end;
    trapq_check_sentinels(sk->tq);
    double pre = sk->gen_steps_pre_active, post = sk->gen_steps_post_active;
    struct move *m = list_first_entry(&sk->tq->moves, struct move, node);
    while (m->print_time + m->move_t + post < t)
        m = list_next_entry(m, node);
    for (;;) {
        double start = m->print_time - pre;
        if (start >= end)
            return end;
        if (drives_stepper(sk, m))
            return start > t ? start : t;
        m = list_next_entry(m, node);
    }
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
        return PG_ERR_FULL;
    struct phasegen_seg *s = &out[(*pnum)++];
    s->count = count;
    s->start_vel = vel;
    s->half_accel = half;
    s->wait_tick = wait_tick;
    return 0;
}

// Emit segments for every tick whose latch time is at or before flush_time.
// Returns the number of entries written to out, PG_ERR_FULL if out is too
// small, or PG_ERR_JUMP if the target moves too far in one tick.
int32_t __visible
phasegen_generate(struct phasegen *pg, double flush_time
                  , struct phasegen_seg *out, int32_t max_out)
{
    int64_t end_tick = (int64_t)floor(
        (flush_time - pg->grid_time) / pg->tick_time) - 1;
    if (!pg->moving) {
        // Sampling every tick of a long rest (a heat-up, a pause) takes long
        // enough to deliver the following motion to the MCU too late, so
        // skip the ticks no move can reach. Tick n samples time
        // grid_time + (n+1)*tick_time.
        double t = pg->grid_time + (pg->last_tick + 2) * pg->tick_time;
        double move_time = next_motion_time(pg->sk, t, flush_time);
        int64_t rest_end = (int64_t)floor(
            (move_time - pg->grid_time) / pg->tick_time) - 2;
        if (rest_end > end_tick)
            rest_end = end_tick;
        if (rest_end > pg->last_tick)
            pg->last_tick = rest_end;
    }
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
                return PG_ERR_FULL;
            pg->moving = 1;
        } else if (still >= IDLE_TICKS || done + still == total) {
            if (still) {
                // A one tick segment at zero velocity leaves the MCU at rest
                if (add_seg(out, &num, max_out, 1, 0, 0, 0))
                    return PG_ERR_FULL;
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
            return PG_ERR_JUMP;
        if (add_seg(out, &num, max_out, count, vel, half, 0))
            return PG_ERR_FULL;
        pg->phase += count * vel + (int64_t)count * count * half;
        done += count;
    }
    pg->last_tick = end_tick;
    return num;
}

void __visible
phasegen_setup_current(struct phasegen *pg, int32_t block_ticks
                       , int32_t lead_blocks, int32_t lag_blocks
                       , int32_t sample_ticks, double current_min
                       , double current_slope, double current_max
                       , double full_scale, int32_t scaler_min
                       , int32_t scaler_max)
{
    pg->block_ticks = block_ticks;
    pg->lead_blocks = lead_blocks;
    pg->lag_blocks = lag_blocks;
    pg->sample_ticks = sample_ticks;
    pg->current_min = current_min;
    pg->current_slope = current_slope;
    pg->current_max = current_max;
    pg->full_scale = full_scale;
    pg->scaler_min = scaler_min;
    pg->scaler_max = scaler_max;
}

// Levels from the block holding first_tick on; the MCU starts at scaler
void __visible
phasegen_start_current(struct phasegen *pg, int64_t first_tick
                       , int32_t scaler)
{
    pg->next_block = first_tick / pg->block_ticks;
    pg->last_scaler = scaler;
}

int32_t __visible
phasegen_current_scaler(struct phasegen *pg, double accel)
{
    double current = pg->current_min + pg->current_slope * fabs(accel);
    if (current > pg->current_max)
        current = pg->current_max;
    int32_t scaler = (int32_t)ceil(256. * current / pg->full_scale - 1e-9);
    if (scaler < pg->scaler_min)
        scaler = pg->scaler_min;
    if (scaler > pg->scaler_max)
        scaler = pg->scaler_max;
    return scaler;
}

static double
tick_print_time(struct phasegen *pg, int64_t tick)
{
    return pg->grid_time + (tick + 1) * pg->tick_time;
}

// Largest |acceleration| (mm/s^2) of the stepper within block b, cached
// for the blocks around it
static double
block_accel(struct phasegen *pg, int64_t b)
{
    int idx = b & (PEAK_CACHE - 1);
    if (pg->peak_block[idx] == b)
        return pg->peak_accel[idx];
    int64_t s = pg->sample_ticks, first = b * pg->block_ticks;
    int64_t last = first + pg->block_ticks - 1;
    double peak = 0.;
    double t_last = tick_print_time(pg, last + s);
    if (next_motion_time(pg->sk, tick_print_time(pg, first - s), t_last)
        < t_last) {
        double dt = s * pg->tick_time, inv = 1. / (dt * dt);
        double p0 = stepper_position(pg, tick_print_time(pg, first - s));
        double p1 = stepper_position(pg, tick_print_time(pg, first));
        int64_t k;
        for (k = first; k <= last; k += s) {
            double p2 = stepper_position(pg, tick_print_time(pg, k + s));
            double a = fabs(p2 - 2. * p1 + p0) * inv;
            if (a > peak)
                peak = a;
            p0 = p1;
            p1 = p2;
        }
    }
    pg->peak_block[idx] = b;
    pg->peak_accel[idx] = peak;
    return peak;
}

// Emit a scaler change for each block that ends at or before end_tick and
// needs a different level than the one before it. Returns the number of
// changes written to out; call again while it fills out.
int32_t __visible
phasegen_current(struct phasegen *pg, int64_t end_tick
                 , struct phasegen_current *out, int32_t max_out)
{
    int64_t bt = pg->block_ticks;
    int64_t last_block = (end_tick + 1) / bt - 1;
    // Moves may have been added since the last call
    int i;
    for (i = 0; i < PEAK_CACHE; i++)
        pg->peak_block[i] = INT64_MIN;
    // Sampling ahead of the planned window must not disturb stepper
    // position tracking for the next phasegen_generate()
    double hold_pos = pg->hold_pos;
    pg->m = NULL;
    int32_t num = 0;
    while (pg->next_block <= last_block && num < max_out) {
        int64_t b = pg->next_block, next = b + 1;
        int64_t first = b - pg->lag_blocks, last = b + pg->lead_blocks;
        double t_first = tick_print_time(pg, first * bt - pg->sample_ticks);
        double t_last = tick_print_time(pg, (last + 1) * bt
                                        + pg->sample_ticks);
        double accel = 0.;
        double t_move = next_motion_time(pg->sk, t_first, t_last);
        if (t_move < t_last) {
            int64_t k;
            for (k = first; k <= last; k++) {
                double a = block_accel(pg, k);
                if (a > accel)
                    accel = a;
            }
        } else {
            // At rest until a later move: skip to the first block whose
            // window may reach it
            double t_end = tick_print_time(pg, (last_block + 1) * bt);
            t_move = next_motion_time(pg->sk, t_last, t_end);
            int64_t tick = (int64_t)floor((t_move - pg->grid_time)
                                          / pg->tick_time) - 1;
            int64_t skip = (tick - pg->sample_ticks) / bt - pg->lead_blocks;
            if (skip > next)
                next = skip;
            if (next > last_block + 1)
                next = last_block + 1;
        }
        int32_t scaler = phasegen_current_scaler(pg, accel);
        if (scaler != pg->last_scaler) {
            out[num].tick = b * bt;
            out[num].scaler = scaler;
            num++;
            pg->last_scaler = scaler;
        }
        pg->next_block = next;
    }
    pg->m = NULL;
    pg->hold_pos = hold_pos;
    return num;
}
