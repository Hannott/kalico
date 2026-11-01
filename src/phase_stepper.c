// Drive stepper driver coil currents directly ("phase stepping")
//
// Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <string.h> // memset
#include "autoconf.h" // CONFIG_PHASE_STEPPING_FREQ
#include "basecmd.h" // oid_alloc
#include "board/gpio.h" // gpio_out_write
#include "board/irq.h" // irq_disable
#include "board/misc.h" // timer_read_time
#include "command.h" // DECL_COMMAND
#include "compiler.h" // barrier
#include "phase_stepper.h" // phase_stepper_update
#include "sched.h" // DECL_SHUTDOWN
#include "spicmds.h" // spidev_oid_lookup

#define TICK_CYCLES (CONFIG_CLOCK_FREQ / CONFIG_PHASE_STEPPING_FREQ)
#define MAX_PHASE_STEPPERS 4
#define MOVE_QUEUE_SIZE 128
#define SPI_WRITE 0x80
#define PAIR_TIMEOUT_US 250

DECL_CONSTANT("PHASE_STEPPER_TICK_FREQ", CONFIG_PHASE_STEPPING_FREQ);
DECL_CONSTANT("PHASE_STEPPER_QUEUE_SIZE", MOVE_QUEUE_SIZE - 1);

// round(32767 * sin(i * pi / 512)) for one quarter of an electrical period
static const uint16_t sine_quarter[257] = {
    0, 201, 402, 603, 804, 1005, 1206, 1407,
    1608, 1809, 2009, 2210, 2410, 2611, 2811, 3012,
    3212, 3412, 3612, 3811, 4011, 4210, 4410, 4609,
    4808, 5007, 5205, 5404, 5602, 5800, 5998, 6195,
    6393, 6590, 6786, 6983, 7179, 7375, 7571, 7767,
    7962, 8157, 8351, 8545, 8739, 8933, 9126, 9319,
    9512, 9704, 9896, 10087, 10278, 10469, 10659, 10849,
    11039, 11228, 11417, 11605, 11793, 11980, 12167, 12353,
    12539, 12725, 12910, 13094, 13279, 13462, 13645, 13828,
    14010, 14191, 14372, 14553, 14732, 14912, 15090, 15269,
    15446, 15623, 15800, 15976, 16151, 16325, 16499, 16673,
    16846, 17018, 17189, 17360, 17530, 17700, 17869, 18037,
    18204, 18371, 18537, 18703, 18868, 19032, 19195, 19357,
    19519, 19680, 19841, 20000, 20159, 20317, 20475, 20631,
    20787, 20942, 21096, 21250, 21403, 21554, 21705, 21856,
    22005, 22154, 22301, 22448, 22594, 22739, 22884, 23027,
    23170, 23311, 23452, 23592, 23731, 23870, 24007, 24143,
    24279, 24413, 24547, 24680, 24811, 24942, 25072, 25201,
    25329, 25456, 25582, 25708, 25832, 25955, 26077, 26198,
    26319, 26438, 26556, 26674, 26790, 26905, 27019, 27133,
    27245, 27356, 27466, 27575, 27683, 27790, 27896, 28001,
    28105, 28208, 28310, 28411, 28510, 28609, 28706, 28803,
    28898, 28992, 29085, 29177, 29268, 29358, 29447, 29534,
    29621, 29706, 29791, 29874, 29956, 30037, 30117, 30195,
    30273, 30349, 30424, 30498, 30571, 30643, 30714, 30783,
    30852, 30919, 30985, 31050, 31113, 31176, 31237, 31297,
    31356, 31414, 31470, 31526, 31580, 31633, 31685, 31736,
    31785, 31833, 31880, 31926, 31971, 32014, 32057, 32098,
    32137, 32176, 32213, 32250, 32285, 32318, 32351, 32382,
    32412, 32441, 32469, 32495, 32521, 32545, 32567, 32589,
    32609, 32628, 32646, 32663, 32678, 32692, 32705, 32717,
    32728, 32737, 32745, 32752, 32757, 32761, 32765, 32766,
    32767,
};

// A constant acceleration segment. Each tick the phase advances by
// vel + half_accel and vel by 2*half_accel, so the integer phase path is
// exactly reversed by a segment with negated start_vel and half_accel.
// An entry with a count of zero is a wait: the following segment starts on
// the tick at the clock stored in start_vel.
struct phase_move {
    uint32_t count;
    int32_t start_vel, half_accel;
};

enum { PF_ARMED = 1, PF_RUNNING = 2 };
enum { PC_SWAP_COILS = 1 };

struct phase_stepper {
    struct spidev_s *spi;
    struct spi_config config;
    struct gpio_out cs;
    uint8_t spi_oid, reg, coil_flags, cs_active_high, flags;
    uint16_t amplitude;
    // A full 32 bit wrap is one electrical period (4 full steps).
    uint32_t phase;
    int32_t vel, half_accel;
    uint32_t count, start_clock, last_sent;
    struct phase_move queue[MOVE_QUEUE_SIZE];
    volatile uint8_t queue_head, queue_tail;
};

static struct phase_stepper *steppers[MAX_PHASE_STEPPERS];
static uint8_t stepper_count, next_stepper, timer_running;
static struct phase_stepper *xfer_owner;
static volatile uint8_t task_spi_active;
// A driver whose message pair from the host is half done (see release)
static struct spidev_s *volatile pair_spi;
static uint32_t pair_start;
static uint8_t xfer_buf[5];
static uint32_t next_tick_clock;

static struct {
    uint32_t ticks, lost_ticks, locked_ticks, pair_timeouts, xfers;
    uint32_t max_cycles;
    uint64_t total_cycles;
} stats;


/****************************************************************
 * Coil currents
 ****************************************************************/

// Signed sine of a 10 bit (1024 per period) phase index, scaled by amp
static int32_t
scaled_sine(uint32_t idx, uint32_t amp)
{
    idx &= 1023;
    uint32_t off = idx & 255;
    uint32_t mag = sine_quarter[idx & 256 ? 256 - off : off];
    int32_t val = (mag * amp + (1 << 14)) >> 15;
    return idx & 512 ? -val : val;
}

// Register value with 9 bit signed coil currents in bits 0-8 and 16-24,
// the direct mode layout of Trinamic drivers.
static uint32_t
coil_value(struct phase_stepper *ps)
{
    uint32_t idx = ps->phase >> 22;
    int32_t sin_val = scaled_sine(idx, ps->amplitude);
    int32_t cos_val = scaled_sine(idx + 256, ps->amplitude);
    if (ps->coil_flags & PC_SWAP_COILS)
        return (cos_val & 0x1ff) | ((uint32_t)(sin_val & 0x1ff) << 16);
    return (sin_val & 0x1ff) | ((uint32_t)(cos_val & 0x1ff) << 16);
}


/****************************************************************
 * SPI transfers
 ****************************************************************/

// Release the chip select of the current transfer (irqs off or in irq)
static void
xfer_finish(void)
{
    uint32_t end = timer_read_time() + timer_from_us(100);
    while (!phase_board_xfer_done())
        if (timer_is_before(end, timer_read_time()))
            shutdown("Phase stepper SPI transfer timeout");
    gpio_out_write(xfer_owner->cs, !xfer_owner->cs_active_high);
    phase_board_xfer_finish();
    xfer_owner = NULL;
}

static void
xfer_start(struct phase_stepper *ps, uint32_t value)
{
    xfer_buf[0] = ps->reg | SPI_WRITE;
    xfer_buf[1] = value >> 24;
    xfer_buf[2] = value >> 16;
    xfer_buf[3] = value >> 8;
    xfer_buf[4] = value;
    spi_prepare(ps->config);
    gpio_out_write(ps->cs, ps->cs_active_high);
    xfer_owner = ps;
    ps->last_sent = value;
    stats.xfers++;
    phase_board_xfer_start(&ps->config, xfer_buf, sizeof(xfer_buf));
}

void
phase_stepper_spi_acquire(void)
{
    if (!timer_running)
        return;
    // May run from shutdown handlers, which run with irqs disabled
    irqstatus_t flag = irq_save();
    task_spi_active = 1;
    if (xfer_owner)
        xfer_finish();
    irq_restore(flag);
}

static int
is_phase_spi(struct spidev_s *spi)
{
    uint32_t i;
    for (i = 0; i < stepper_count; i++)
        if (steppers[i]->spi == spi && steppers[i]->flags)
            return 1;
    return 0;
}

// Trinamic drivers answer a read in the following message, so the host
// always sends them messages in pairs (read and read again, or write and
// a verifying read). A coil write between the two would make the second
// answer return the coil register, so the bus stays locked until the
// second message of a pair to a phase stepped driver.
void
phase_stepper_spi_release(struct spidev_s *spi)
{
    irqstatus_t flag = irq_save();
    if (spi == pair_spi) {
        pair_spi = NULL;
    } else if (!pair_spi && timer_running && is_phase_spi(spi)) {
        pair_spi = spi;
        pair_start = timer_read_time();
    }
    task_spi_active = 0;
    irq_restore(flag);
}


/****************************************************************
 * Updates
 ****************************************************************/

// Advance by the tick at clock now
static void
stepper_tick(struct phase_stepper *ps, uint32_t now)
{
    while (!ps->count) {
        uint8_t tail = ps->queue_tail;
        // Motion always ends at rest, so running out of segments or
        // reaching a wait while moving means the host fell behind.
        if (tail == ps->queue_head) {
            if (ps->vel)
                shutdown("Phase stepper queue underrun");
            return;
        }
        struct phase_move *m = &ps->queue[tail];
        if (!m->count) {
            if (ps->vel)
                shutdown("Phase stepper queue underrun");
            uint32_t clock = m->start_vel;
            if (timer_is_before(now, clock))
                return;
            if (!timer_is_before(now, clock + TICK_CYCLES))
                shutdown("Phase stepper move too late");
        } else {
            ps->count = m->count;
            ps->vel = m->start_vel;
            ps->half_accel = m->half_accel;
        }
        ps->queue_tail = (tail + 1) % MOVE_QUEUE_SIZE;
    }
    ps->phase += ps->vel + ps->half_accel;
    ps->vel += 2 * ps->half_accel;
    ps->count--;
}

void
phase_stepper_update(void)
{
    uint32_t start = timer_read_time();

    // Normally one tick has passed; more if irqs were disabled for long.
    uint32_t ticks = 1;
    if (!timer_is_before(start, next_tick_clock))
        ticks += (start - next_tick_clock) / TICK_CYCLES;
    uint32_t now = next_tick_clock + (ticks - 1) * TICK_CYCLES;
    next_tick_clock += ticks * TICK_CYCLES;
    stats.ticks += ticks;
    stats.lost_ticks += ticks - 1;

    uint32_t i, t;
    for (i = 0; i < stepper_count; i++) {
        struct phase_stepper *ps = steppers[i];
        if (ps->flags & PF_ARMED && !timer_is_before(now, ps->start_clock))
            ps->flags = PF_RUNNING;
        if (!(ps->flags & PF_RUNNING))
            continue;
        for (t = 0; t < ticks; t++)
            stepper_tick(ps, now - (ticks - 1 - t) * TICK_CYCLES);
    }

    if (pair_spi && !task_spi_active && timer_is_before(
            pair_start + timer_from_us(PAIR_TIMEOUT_US), start)) {
        pair_spi = NULL;
        stats.pair_timeouts++;
    }
    if (task_spi_active || pair_spi) {
        stats.locked_ticks++;
    } else {
        // The previous transfer's chip select is released here, so new
        // currents latch on a tick boundary, one tick after computing.
        if (xfer_owner)
            xfer_finish();
        for (i = 0; i < stepper_count; i++) {
            struct phase_stepper *ps = steppers[next_stepper];
            next_stepper = (next_stepper + 1) % stepper_count;
            if (!(ps->flags & (PF_ARMED | PF_RUNNING)))
                continue;
            uint32_t value = coil_value(ps);
            if (value != ps->last_sent) {
                xfer_start(ps, value);
                break;
            }
        }
    }

    uint32_t cycles = timer_read_time() - start;
    stats.total_cycles += cycles;
    if (cycles > stats.max_cycles)
        stats.max_cycles = cycles;
}

static void
timer_start(void)
{
    if (timer_running)
        return;
    irq_disable();
    next_tick_clock = timer_read_time() + TICK_CYCLES;
    timer_running = 1;
    irq_enable();
    phase_board_timer_start();
}

static void
timer_stop(void)
{
    phase_board_timer_stop();
    irq_disable();
    if (xfer_owner)
        xfer_finish();
    pair_spi = NULL;
    timer_running = 0;
    irq_enable();
}


/****************************************************************
 * Commands
 ****************************************************************/

void
command_config_phase_stepper(uint32_t *args)
{
    if (stepper_count >= ARRAY_SIZE(steppers))
        shutdown("Too many phase steppers");
    struct phase_stepper *ps = oid_alloc(
        args[0], command_config_phase_stepper, sizeof(*ps));
    ps->spi_oid = args[1];
    ps->reg = args[2];
    ps->coil_flags = args[3];
    steppers[stepper_count++] = ps;
}
DECL_COMMAND(command_config_phase_stepper,
             "config_phase_stepper oid=%c spi_oid=%c reg=%c coil_flags=%c");

static struct phase_stepper *
phase_stepper_oid_lookup(uint8_t oid)
{
    return oid_lookup(oid, command_config_phase_stepper);
}

static void
queue_push(struct phase_stepper *ps, uint32_t count, int32_t start_vel
           , int32_t half_accel)
{
    uint8_t head = ps->queue_head, next = (head + 1) % MOVE_QUEUE_SIZE;
    if (next == ps->queue_tail)
        shutdown("Phase stepper move queue overflow");
    struct phase_move *m = &ps->queue[head];
    m->count = count;
    m->start_vel = start_vel;
    m->half_accel = half_accel;
    barrier();
    ps->queue_head = next;
}

void
command_phase_stepper_queue_move(uint32_t *args)
{
    struct phase_stepper *ps = phase_stepper_oid_lookup(args[0]);
    if (!args[1])
        shutdown("Invalid phase stepper move count");
    queue_push(ps, args[1], args[2], args[3]);
}
DECL_COMMAND(command_phase_stepper_queue_move,
             "phase_stepper_queue_move oid=%c count=%u start_vel=%i"
             " half_accel=%i");

void
command_phase_stepper_queue_wait(uint32_t *args)
{
    struct phase_stepper *ps = phase_stepper_oid_lookup(args[0]);
    queue_push(ps, 0, args[1], 0);
}
DECL_COMMAND(command_phase_stepper_queue_wait,
             "phase_stepper_queue_wait oid=%c clock=%u");

void
command_phase_stepper_start(uint32_t *args)
{
    struct phase_stepper *ps = phase_stepper_oid_lookup(args[0]);
    struct spidev_s *spi = spidev_oid_lookup(ps->spi_oid);
    struct spi_config config;
    uint8_t cs_active_high;
    if (!spidev_get_hardware_config(spi, &config, &cs_active_high))
        shutdown("Phase stepping requires hardware SPI with a CS pin");
    phase_board_xfer_setup(&config);
    irq_disable();
    ps->spi = spi;
    ps->config = config;
    ps->cs = spidev_get_cs_pin(spi);
    ps->cs_active_high = cs_active_high;
    ps->start_clock = args[1];
    ps->phase = args[2];
    ps->amplitude = args[3] > 255 ? 255 : args[3];
    ps->vel = ps->half_accel = ps->count = 0;
    ps->last_sent = ~coil_value(ps);
    ps->flags = PF_ARMED;
    irq_enable();
    timer_start();
}
DECL_COMMAND(command_phase_stepper_start,
             "phase_stepper_start oid=%c clock=%u phase=%u amplitude=%hu");

void
command_phase_stepper_stop(uint32_t *args)
{
    struct phase_stepper *ps = phase_stepper_oid_lookup(args[0]);
    irq_disable();
    if (xfer_owner == ps)
        xfer_finish();
    ps->flags = 0;
    ps->count = 0;
    ps->queue_tail = ps->queue_head;
    irq_enable();
    uint32_t i;
    for (i = 0; i < stepper_count; i++)
        if (steppers[i]->flags)
            return;
    if (timer_running)
        timer_stop();
}
DECL_COMMAND(command_phase_stepper_stop, "phase_stepper_stop oid=%c");

void
command_phase_stepper_query(uint32_t *args)
{
    uint8_t oid = args[0];
    struct phase_stepper *ps = phase_stepper_oid_lookup(oid);
    irq_disable();
    uint32_t phase = ps->phase, flags = ps->flags;
    uint32_t tick_clock = next_tick_clock;
    uint32_t pending = ps->count || ps->queue_tail != ps->queue_head;
    uint32_t ticks = stats.ticks, lost = stats.lost_ticks;
    uint32_t locked = stats.locked_ticks, max_cycles = stats.max_cycles;
    uint32_t xfers = stats.xfers, pair_timeouts = stats.pair_timeouts;
    uint64_t total_cycles = stats.total_cycles;
    memset(&stats, 0, sizeof(stats));
    irq_enable();
    uint32_t avg_cycles = ticks ? total_cycles / ticks : 0;
    sendf("phase_stepper_state oid=%c phase=%u flags=%c pending=%c ticks=%u"
          " lost_ticks=%u locked_ticks=%u pair_timeouts=%u xfers=%u"
          " max_cycles=%u avg_cycles=%u tick_clock=%u", oid, phase, flags
          , pending, ticks, lost, locked, pair_timeouts, xfers, max_cycles
          , avg_cycles, tick_clock);
}
DECL_COMMAND(command_phase_stepper_query, "phase_stepper_query oid=%c");

void
phase_stepper_shutdown(void)
{
    if (!timer_running)
        return;
    phase_board_timer_stop();
    if (xfer_owner) {
        phase_board_xfer_abort();
        gpio_out_write(xfer_owner->cs, !xfer_owner->cs_active_high);
        xfer_owner = NULL;
    }
    uint32_t i;
    for (i = 0; i < stepper_count; i++)
        steppers[i]->flags = 0;
    task_spi_active = 0;
    pair_spi = NULL;
    timer_running = 0;
}
DECL_SHUTDOWN(phase_stepper_shutdown);
