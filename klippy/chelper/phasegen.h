#ifndef PHASEGEN_H
#define PHASEGEN_H

#include <stdint.h> // int64_t

// A segment for the MCU's phase stepper queue, or (count == 0) a wait
// until the update at wait_tick on the MCU's update grid
struct phasegen_seg {
    uint32_t count;
    int32_t start_vel, half_accel;
    int64_t wait_tick;
};

enum { PG_ERR_FULL = -1, PG_ERR_JUMP = -2 };

// From the update at tick on, the driver's current scaler is scaler
struct phasegen_current {
    int64_t tick;
    int32_t scaler;
};

struct stepper_kinematics;
struct phasegen *phasegen_alloc(void);
void phasegen_free(struct phasegen *pg);
void phasegen_setup(struct phasegen *pg, struct stepper_kinematics *sk
                    , double tick_time, double phase_per_mm
                    , double tolerance);
void phasegen_start(struct phasegen *pg, double grid_time, int64_t start_tick
                    , int64_t start_phase, double ref_pos);
int32_t phasegen_generate(struct phasegen *pg, double flush_time
                          , struct phasegen_seg *out, int32_t max_out);
void phasegen_rebase(struct phasegen *pg, double ref_pos);
double phasegen_target(struct phasegen *pg, int64_t tick);
int64_t phasegen_get_phase(struct phasegen *pg);
int64_t phasegen_get_last_tick(struct phasegen *pg);
void phasegen_setup_current(struct phasegen *pg, int32_t block_ticks
                            , int32_t lead_blocks, int32_t lag_blocks
                            , int32_t sample_ticks, double current_min
                            , double current_slope, double current_max
                            , double full_scale, int32_t scaler_min
                            , int32_t scaler_max);
void phasegen_start_current(struct phasegen *pg, int64_t first_tick
                            , int32_t scaler);
int32_t phasegen_current_scaler(struct phasegen *pg, double accel);
int32_t phasegen_current(struct phasegen *pg, int64_t end_tick
                         , struct phasegen_current *out, int32_t max_out);

#endif // phasegen.h
