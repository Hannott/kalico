// Phase stepping updates from the scheduler with blocking SPI transfers
//
// Works on any board with hardware SPI, at a lower update rate than a
// board specific timer and DMA implementation.
//
// Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include "autoconf.h" // CONFIG_PHASE_STEPPING_FREQ
#include "board/gpio.h" // spi_transfer
#include "board/misc.h" // timer_read_time
#include "phase_stepper.h" // phase_board_timer_start
#include "sched.h" // sched_add_timer

#define TICK_CYCLES (CONFIG_CLOCK_FREQ / CONFIG_PHASE_STEPPING_FREQ)

static struct timer update_timer;

static uint_fast8_t
update_event(struct timer *t)
{
    phase_stepper_update();
    t->waketime += TICK_CYCLES;
    return SF_RESCHEDULE;
}

void
phase_board_timer_start(void)
{
    sched_del_timer(&update_timer);
    update_timer.func = update_event;
    update_timer.waketime = timer_read_time() + TICK_CYCLES;
    sched_add_timer(&update_timer);
}

void
phase_board_timer_stop(void)
{
    sched_del_timer(&update_timer);
}

void
phase_board_xfer_setup(struct spi_config *config)
{
}

void
phase_board_xfer_start(struct spi_config *config, uint8_t *data
                       , uint_fast8_t len)
{
    spi_transfer(*config, 0, len, data);
}

int
phase_board_xfer_done(void)
{
    return 1;
}

void
phase_board_xfer_finish(void)
{
}

void
phase_board_xfer_abort(void)
{
}
