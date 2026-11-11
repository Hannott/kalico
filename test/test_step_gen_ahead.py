import os
import time

import pytest

from klippy import chelper

MCU_FREQ = 50000000.0
STEP_DIST = 0.0125


class StepperRig:
    """A single cartesian stepper driven through steppersync"""

    def __init__(self):
        self.ffi, self.lib = ffi, lib = chelper.get_ffi()
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.sq = lib.serialqueue_alloc(self.devnull, b"f", 0, b"test")
        self.ssm = lib.steppersyncmgr_alloc()
        self.ss = lib.steppersyncmgr_alloc_steppersync(self.ssm)
        self.se = lib.steppersync_alloc_syncemitter(self.ss, b"stepper_x", 1)
        self.sc = lib.syncemitter_get_stepcompress(self.se)
        lib.stepcompress_fill(self.sc, 0, int(25e-6 * MCU_FREQ), 1, 2)
        lib.stepcompress_reset(self.sc, 0)
        lib.stepcompress_set_last_position(self.sc, 0, 0)
        self.tq = lib.trapq_alloc()
        self.sk = lib.cartesian_stepper_alloc(b"x")
        lib.itersolve_set_trapq(self.sk, self.tq, STEP_DIST)
        lib.itersolve_set_position(self.sk, 0.0, 0.0, 0.0)
        lib.syncemitter_set_stepper_kinematics(self.se, self.sk)
        lib.steppersync_setup_movequeue(self.ss, self.sq, 500)
        lib.steppersync_set_time(self.ss, 0.0, MCU_FREQ)

    def sync_mcu_clock(self, print_time):
        # Make the serialqueue estimate that the mcu is at print_time now
        now = self.lib.get_monotonic()
        clock = int(print_time * MCU_FREQ)
        self.lib.serialqueue_set_clock_est(self.sq, MCU_FREQ, now, clock, clock)

    def cruise(self, print_time, duration, velocity):
        self.lib.trapq_append(
            self.tq,
            print_time,
            0.0,
            duration,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            velocity,
            velocity,
            0.0,
        )

    def gen_steps(self, flush_time, convert_time, gen_time):
        return self.lib.steppersyncmgr_gen_steps(
            self.ssm, flush_time, convert_time, gen_time, 0.0
        )

    def last_queued_clock(self):
        hist = self.ffi.new("struct pull_history_steps[1]")
        count = self.lib.stepcompress_extract_old(self.sc, hist, 1, 0, 2**63)
        return hist[0].last_clock if count else 0

    def queued_steps(self):
        hist = self.ffi.new("struct pull_history_steps[100000]")
        count = self.lib.stepcompress_extract_old(
            self.sc, hist, len(hist), 0, 2**63
        )
        return sum(abs(hist[i].step_count) for i in range(count))

    def close(self):
        lib = self.lib
        lib.steppersyncmgr_set_tx_lead(self.ssm, 0.0)
        lib.steppersync_detach_movequeue(self.ss)
        lib.steppersyncmgr_free(self.ssm)
        lib.serialqueue_exit(self.sq)
        lib.serialqueue_free(self.sq)
        lib.trapq_free(self.tq)
        os.close(self.devnull)


@pytest.fixture
def rig():
    r = StepperRig()
    yield r
    r.close()


def test_steps_beyond_convert_time_are_held(rig):
    rig.cruise(1.0, 3.0, 100.0)
    assert rig.gen_steps(1.2, 1.25, 3.5) == 0
    # Only converted steps reach the mcu clock queue and history
    assert rig.last_queued_clock() <= 1.25 * MCU_FREQ
    assert rig.queued_steps() < 0.3 * 100.0 / STEP_DIST
    # Converting the rest later yields every generated step
    assert rig.gen_steps(4.0, 4.0, 4.0) == 0
    assert rig.queued_steps() == int(3.0 * 100.0 / STEP_DIST)


def test_background_transmit_keeps_feeding_while_host_stalls(rig):
    rig.cruise(1.0, 3.0, 100.0)
    rig.sync_mcu_clock(1.0)
    assert rig.gen_steps(1.3, 1.35, 3.5) == 0
    before = rig.last_queued_clock()
    rig.lib.steppersyncmgr_set_tx_lead(rig.ssm, 0.25)
    # Simulate a stalled host: no further gen_steps() calls
    time.sleep(0.6)
    after = rig.last_queued_clock()
    assert before <= 1.35 * MCU_FREQ
    # The mcu is at ~1.6 now; the thread keeps it supplied 0.25s ahead
    assert after >= 1.8 * MCU_FREQ
    assert rig.gen_steps(4.0, 4.0, 4.0) == 0
    assert rig.queued_steps() == int(3.0 * 100.0 / STEP_DIST)


def test_background_transmit_stops_at_generated_steps(rig):
    rig.cruise(1.0, 3.0, 100.0)
    rig.sync_mcu_clock(1.0)
    assert rig.gen_steps(1.1, 1.15, 1.6) == 0
    rig.lib.steppersyncmgr_set_tx_lead(rig.ssm, 0.25)
    time.sleep(0.8)
    # Steps after gen_time - 0.050 still need their compression window
    assert rig.last_queued_clock() <= 1.6 * MCU_FREQ


def test_background_transmit_is_idle_when_disabled(rig):
    rig.cruise(1.0, 3.0, 100.0)
    rig.sync_mcu_clock(1.0)
    assert rig.gen_steps(1.1, 1.15, 3.5) == 0
    before = rig.last_queued_clock()
    time.sleep(0.3)
    assert rig.last_queued_clock() == before


def test_detached_movequeue_discards_messages(rig):
    rig.cruise(1.0, 1.0, 100.0)
    rig.lib.steppersync_detach_movequeue(rig.ss)
    assert rig.gen_steps(2.5, 2.5, 2.5) == 0
