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
double phasegen_target(struct phasegen *pg, int64_t tick);
int64_t phasegen_get_phase(struct phasegen *pg);
int64_t phasegen_get_last_tick(struct phasegen *pg);

#endif // phasegen.h
