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
#define HOST_MSG_SIZE 5

DECL_CONSTANT("PHASE_STEPPER_TICK_FREQ", CONFIG_PHASE_STEPPING_FREQ);
DECL_CONSTANT("PHASE_STEPPER_QUEUE_SIZE", MOVE_QUEUE_SIZE - 1);

// round(32767 * sin(i * pi / 2048)) for one quarter of an electrical period
static const uint16_t sine_quarter[1025] = {
    0, 50, 101, 151, 201, 251, 302, 352,
    402, 452, 503, 553, 603, 653, 704, 754,
    804, 854, 905, 955, 1005, 1055, 1106, 1156,
    1206, 1256, 1307, 1357, 1407, 1457, 1507, 1558,
    1608, 1658, 1708, 1758, 1809, 1859, 1909, 1959,
    2009, 2059, 2110, 2160, 2210, 2260, 2310, 2360,
    2410, 2461, 2511, 2561, 2611, 2661, 2711, 2761,
    2811, 2861, 2911, 2962, 3012, 3062, 3112, 3162,
    3212, 3262, 3312, 3362, 3412, 3462, 3512, 3562,
    3612, 3662, 3712, 3761, 3811, 3861, 3911, 3961,
    4011, 4061, 4111, 4161, 4210, 4260, 4310, 4360,
    4410, 4460, 4509, 4559, 4609, 4659, 4708, 4758,
    4808, 4858, 4907, 4957, 5007, 5056, 5106, 5156,
    5205, 5255, 5305, 5354, 5404, 5453, 5503, 5552,
    5602, 5651, 5701, 5750, 5800, 5849, 5899, 5948,
    5998, 6047, 6096, 6146, 6195, 6245, 6294, 6343,
    6393, 6442, 6491, 6540, 6590, 6639, 6688, 6737,
    6786, 6836, 6885, 6934, 6983, 7032, 7081, 7130,
    7179, 7228, 7277, 7326, 7375, 7424, 7473, 7522,
    7571, 7620, 7669, 7718, 7767, 7815, 7864, 7913,
    7962, 8010, 8059, 8108, 8157, 8205, 8254, 8303,
    8351, 8400, 8448, 8497, 8545, 8594, 8642, 8691,
    8739, 8788, 8836, 8885, 8933, 8981, 9030, 9078,
    9126, 9175, 9223, 9271, 9319, 9367, 9416, 9464,
    9512, 9560, 9608, 9656, 9704, 9752, 9800, 9848,
    9896, 9944, 9992, 10039, 10087, 10135, 10183, 10231,
    10278, 10326, 10374, 10421, 10469, 10517, 10564, 10612,
    10659, 10707, 10754, 10802, 10849, 10897, 10944, 10992,
    11039, 11086, 11133, 11181, 11228, 11275, 11322, 11370,
    11417, 11464, 11511, 11558, 11605, 11652, 11699, 11746,
    11793, 11840, 11886, 11933, 11980, 12027, 12074, 12120,
    12167, 12214, 12260, 12307, 12353, 12400, 12446, 12493,
    12539, 12586, 12632, 12679, 12725, 12771, 12817, 12864,
    12910, 12956, 13002, 13048, 13094, 13141, 13187, 13233,
    13279, 13324, 13370, 13416, 13462, 13508, 13554, 13599,
    13645, 13691, 13736, 13782, 13828, 13873, 13919, 13964,
    14010, 14055, 14101, 14146, 14191, 14236, 14282, 14327,
    14372, 14417, 14462, 14507, 14553, 14598, 14643, 14688,
    14732, 14777, 14822, 14867, 14912, 14956, 15001, 15046,
    15090, 15135, 15180, 15224, 15269, 15313, 15358, 15402,
    15446, 15491, 15535, 15579, 15623, 15667, 15712, 15756,
    15800, 15844, 15888, 15932, 15976, 16019, 16063, 16107,
    16151, 16195, 16238, 16282, 16325, 16369, 16413, 16456,
    16499, 16543, 16586, 16630, 16673, 16716, 16759, 16802,
    16846, 16889, 16932, 16975, 17018, 17061, 17104, 17146,
    17189, 17232, 17275, 17317, 17360, 17403, 17445, 17488,
    17530, 17573, 17615, 17657, 17700, 17742, 17784, 17827,
    17869, 17911, 17953, 17995, 18037, 18079, 18121, 18163,
    18204, 18246, 18288, 18330, 18371, 18413, 18454, 18496,
    18537, 18579, 18620, 18661, 18703, 18744, 18785, 18826,
    18868, 18909, 18950, 18991, 19032, 19072, 19113, 19154,
    19195, 19236, 19276, 19317, 19357, 19398, 19438, 19479,
    19519, 19560, 19600, 19640, 19680, 19721, 19761, 19801,
    19841, 19881, 19921, 19961, 20000, 20040, 20080, 20120,
    20159, 20199, 20238, 20278, 20317, 20357, 20396, 20436,
    20475, 20514, 20553, 20592, 20631, 20670, 20709, 20748,
    20787, 20826, 20865, 20904, 20942, 20981, 21019, 21058,
    21096, 21135, 21173, 21212, 21250, 21288, 21326, 21364,
    21403, 21441, 21479, 21516, 21554, 21592, 21630, 21668,
    21705, 21743, 21781, 21818, 21856, 21893, 21930, 21968,
    22005, 22042, 22079, 22116, 22154, 22191, 22227, 22264,
    22301, 22338, 22375, 22411, 22448, 22485, 22521, 22558,
    22594, 22631, 22667, 22703, 22739, 22776, 22812, 22848,
    22884, 22920, 22956, 22991, 23027, 23063, 23099, 23134,
    23170, 23205, 23241, 23276, 23311, 23347, 23382, 23417,
    23452, 23487, 23522, 23557, 23592, 23627, 23662, 23697,
    23731, 23766, 23801, 23835, 23870, 23904, 23938, 23973,
    24007, 24041, 24075, 24109, 24143, 24177, 24211, 24245,
    24279, 24312, 24346, 24380, 24413, 24447, 24480, 24514,
    24547, 24580, 24613, 24647, 24680, 24713, 24746, 24779,
    24811, 24844, 24877, 24910, 24942, 24975, 25007, 25040,
    25072, 25105, 25137, 25169, 25201, 25233, 25265, 25297,
    25329, 25361, 25393, 25425, 25456, 25488, 25519, 25551,
    25582, 25614, 25645, 25676, 25708, 25739, 25770, 25801,
    25832, 25863, 25893, 25924, 25955, 25986, 26016, 26047,
    26077, 26108, 26138, 26168, 26198, 26229, 26259, 26289,
    26319, 26349, 26378, 26408, 26438, 26468, 26497, 26527,
    26556, 26586, 26615, 26644, 26674, 26703, 26732, 26761,
    26790, 26819, 26848, 26876, 26905, 26934, 26962, 26991,
    27019, 27048, 27076, 27104, 27133, 27161, 27189, 27217,
    27245, 27273, 27300, 27328, 27356, 27384, 27411, 27439,
    27466, 27493, 27521, 27548, 27575, 27602, 27629, 27656,
    27683, 27710, 27737, 27764, 27790, 27817, 27843, 27870,
    27896, 27923, 27949, 27975, 28001, 28027, 28053, 28079,
    28105, 28131, 28157, 28182, 28208, 28234, 28259, 28284,
    28310, 28335, 28360, 28385, 28411, 28436, 28460, 28485,
    28510, 28535, 28560, 28584, 28609, 28633, 28658, 28682,
    28706, 28730, 28755, 28779, 28803, 28827, 28850, 28874,
    28898, 28922, 28945, 28969, 28992, 29016, 29039, 29062,
    29085, 29108, 29131, 29154, 29177, 29200, 29223, 29246,
    29268, 29291, 29313, 29336, 29358, 29380, 29403, 29425,
    29447, 29469, 29491, 29513, 29534, 29556, 29578, 29599,
    29621, 29642, 29664, 29685, 29706, 29728, 29749, 29770,
    29791, 29812, 29832, 29853, 29874, 29894, 29915, 29936,
    29956, 29976, 29997, 30017, 30037, 30057, 30077, 30097,
    30117, 30136, 30156, 30176, 30195, 30215, 30234, 30253,
    30273, 30292, 30311, 30330, 30349, 30368, 30387, 30406,
    30424, 30443, 30462, 30480, 30498, 30517, 30535, 30553,
    30571, 30589, 30607, 30625, 30643, 30661, 30679, 30696,
    30714, 30731, 30749, 30766, 30783, 30800, 30818, 30835,
    30852, 30868, 30885, 30902, 30919, 30935, 30952, 30968,
    30985, 31001, 31017, 31033, 31050, 31066, 31082, 31097,
    31113, 31129, 31145, 31160, 31176, 31191, 31206, 31222,
    31237, 31252, 31267, 31282, 31297, 31312, 31327, 31341,
    31356, 31371, 31385, 31400, 31414, 31428, 31442, 31456,
    31470, 31484, 31498, 31512, 31526, 31539, 31553, 31567,
    31580, 31593, 31607, 31620, 31633, 31646, 31659, 31672,
    31685, 31698, 31710, 31723, 31736, 31748, 31760, 31773,
    31785, 31797, 31809, 31821, 31833, 31845, 31857, 31869,
    31880, 31892, 31903, 31915, 31926, 31937, 31949, 31960,
    31971, 31982, 31993, 32004, 32014, 32025, 32036, 32046,
    32057, 32067, 32077, 32087, 32098, 32108, 32118, 32128,
    32137, 32147, 32157, 32166, 32176, 32185, 32195, 32204,
    32213, 32223, 32232, 32241, 32250, 32258, 32267, 32276,
    32285, 32293, 32302, 32310, 32318, 32327, 32335, 32343,
    32351, 32359, 32367, 32375, 32382, 32390, 32397, 32405,
    32412, 32420, 32427, 32434, 32441, 32448, 32455, 32462,
    32469, 32476, 32482, 32489, 32495, 32502, 32508, 32514,
    32521, 32527, 32533, 32539, 32545, 32550, 32556, 32562,
    32567, 32573, 32578, 32584, 32589, 32594, 32599, 32604,
    32609, 32614, 32619, 32624, 32628, 32633, 32637, 32642,
    32646, 32650, 32655, 32659, 32663, 32667, 32671, 32674,
    32678, 32682, 32685, 32689, 32692, 32696, 32699, 32702,
    32705, 32708, 32711, 32714, 32717, 32720, 32722, 32725,
    32728, 32730, 32732, 32735, 32737, 32739, 32741, 32743,
    32745, 32747, 32748, 32750, 32752, 32753, 32755, 32756,
    32757, 32758, 32759, 32760, 32761, 32762, 32763, 32764,
    32765, 32765, 32766, 32766, 32766, 32767, 32767, 32767,
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
    // Coil current scale: run_amp while moving. After rest_ticks at rest it
    // steps down by one every hold_ramp ticks to hold_amp, as a driver
    // lowers its current to the hold current at standstill.
    uint16_t amplitude, run_amp, hold_amp;
    uint32_t hold_delay, hold_ramp, rest_ticks;
    // Motor direction (0 forward, 1 backward) and per direction angle
    // corrections in 1/4096 of an electrical period, by phase >> 22
    uint8_t dir;
    int8_t correction[2][1024];
    // A full 32 bit wrap is one electrical period (4 full steps).
    uint32_t phase;
    int32_t vel, half_accel;
    uint32_t count, start_clock, last_sent;
    // The host's last message to the driver, and whether a coil write
    // went to the driver after it
    uint8_t host_msg[HOST_MSG_SIZE], host_msg_len, coil_written;
    struct phase_move queue[MOVE_QUEUE_SIZE];
    volatile uint8_t queue_head, queue_tail;
};

static struct phase_stepper *steppers[MAX_PHASE_STEPPERS];
static uint8_t stepper_count, next_stepper, timer_running;
static struct phase_stepper *xfer_owner;
static volatile uint8_t task_spi_active;
static uint8_t xfer_buf[5];
static uint32_t next_tick_clock;

static struct {
    uint32_t ticks, lost_ticks, locked_ticks, replays, xfers;
    uint32_t max_cycles;
    uint64_t total_cycles;
} stats;


/****************************************************************
 * Coil currents
 ****************************************************************/

// Signed sine of a 12 bit (4096 per period) angle, scaled by amp
static int32_t
scaled_sine(uint32_t angle, uint32_t amp)
{
    angle &= 4095;
    uint32_t off = angle & 1023;
    uint32_t mag = sine_quarter[angle & 1024 ? 1024 - off : off];
    int32_t val = (mag * amp + (1 << 14)) >> 15;
    return angle & 2048 ? -val : val;
}

// Register value with 9 bit signed coil currents in bits 0-8 and 16-24,
// the direct mode layout of Trinamic drivers.
static uint32_t
coil_value(struct phase_stepper *ps)
{
    // The motor's own position error is cancelled by shifting the current
    // angle by a correction that depends on the angle and direction.
    uint32_t angle = ((ps->phase >> 20)
                      + ps->correction[ps->dir][ps->phase >> 22]);
    int32_t sin_val = scaled_sine(angle, ps->amplitude);
    int32_t cos_val = scaled_sine(angle + 1024, ps->amplitude);
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
    ps->coil_written = 1;
    stats.xfers++;
    phase_board_xfer_start(&ps->config, xfer_buf, sizeof(xfer_buf));
}

static struct phase_stepper *
spi_stepper(struct spidev_s *spi)
{
    uint32_t i;
    for (i = 0; i < stepper_count; i++)
        if (steppers[i]->spi == spi)
            return steppers[i];
    return NULL;
}

// Trinamic drivers answer each message with the register that the message
// before it addressed, so the host reads a register with two messages. A
// coil write between the two would make the second answer return the coil
// register. However long the host takes between its messages, the answer
// it reads stays right if its previous message is sent again first.
static void
replay_host_msg(struct phase_stepper *ps)
{
    uint8_t msg[HOST_MSG_SIZE];
    memcpy(msg, ps->host_msg, ps->host_msg_len);
    spi_prepare(ps->config);
    gpio_out_write(ps->cs, ps->cs_active_high);
    spi_transfer(ps->config, 0, ps->host_msg_len, msg);
    gpio_out_write(ps->cs, !ps->cs_active_high);
    stats.replays++;
}

void
phase_stepper_spi_acquire(struct spidev_s *spi, uint8_t receive_data
                          , uint8_t data_len, uint8_t *data)
{
    if (timer_running) {
        // May run from shutdown handlers, which run with irqs disabled
        irqstatus_t flag = irq_save();
        task_spi_active = 1;
        if (xfer_owner)
            xfer_finish();
        irq_restore(flag);
    }
    struct phase_stepper *ps = spi_stepper(spi);
    if (!ps)
        return;
    if (receive_data && ps->coil_written && ps->host_msg_len)
        replay_host_msg(ps);
    ps->coil_written = 0;
    ps->host_msg_len = data_len <= HOST_MSG_SIZE ? data_len : 0;
    memcpy(ps->host_msg, data, ps->host_msg_len);
}

void
phase_stepper_spi_release(void)
{
    task_spi_active = 0;
}


/****************************************************************
 * Updates
 ****************************************************************/

// A tick with the motor at rest
static void
stepper_rest(struct phase_stepper *ps)
{
    if (ps->amplitude <= ps->hold_amp)
        return;
    if (ps->rest_ticks) {
        ps->rest_ticks--;
        return;
    }
    ps->amplitude--;
    ps->rest_ticks = ps->hold_ramp;
}

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
            stepper_rest(ps);
            return;
        }
        struct phase_move *m = &ps->queue[tail];
        if (!m->count) {
            if (ps->vel)
                shutdown("Phase stepper queue underrun");
            uint32_t clock = m->start_vel;
            if (timer_is_before(now, clock)) {
                stepper_rest(ps);
                return;
            }
            if (!timer_is_before(now, clock + TICK_CYCLES))
                shutdown("Phase stepper move too late");
        } else {
            ps->count = m->count;
            ps->vel = m->start_vel;
            ps->half_accel = m->half_accel;
            ps->amplitude = ps->run_amp;
            ps->rest_ticks = ps->hold_delay;
        }
        ps->queue_tail = (tail + 1) % MOVE_QUEUE_SIZE;
    }
    int32_t step = ps->vel + ps->half_accel;
    if (step)
        ps->dir = step < 0;
    ps->phase += step;
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

    if (task_spi_active) {
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
command_phase_stepper_set_correction(uint32_t *args)
{
    struct phase_stepper *ps = phase_stepper_oid_lookup(args[0]);
    uint32_t dir = args[1], offset = args[2], len = args[3];
    uint8_t *data = command_decode_ptr(args[4]);
    if (dir > 1 || offset + len > ARRAY_SIZE(ps->correction[0]))
        shutdown("Invalid phase stepper correction");
    irq_disable();
    memcpy(&ps->correction[dir][offset], data, len);
    irq_enable();
}
DECL_COMMAND(command_phase_stepper_set_correction,
             "phase_stepper_set_correction oid=%c dir=%c offset=%hu"
             " data=%*s");

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
    ps->amplitude = ps->run_amp = ps->hold_amp = args[3] > 255 ? 255 : args[3];
    ps->rest_ticks = 0;
    ps->vel = ps->half_accel = ps->count = 0;
    ps->dir = 0;
    ps->last_sent = ~coil_value(ps);
    ps->flags = PF_ARMED;
    irq_enable();
    timer_start();
}
DECL_COMMAND(command_phase_stepper_start,
             "phase_stepper_start oid=%c clock=%u phase=%u amplitude=%hu");

// Hold current scale and the timing of the step down to it, sent after
// phase_stepper_start and whenever the driver's currents change
void
command_phase_stepper_set_hold(uint32_t *args)
{
    struct phase_stepper *ps = phase_stepper_oid_lookup(args[0]);
    uint32_t hold_amp = args[1] < ps->run_amp ? args[1] : ps->run_amp;
    irq_disable();
    ps->hold_amp = hold_amp;
    ps->hold_delay = ps->rest_ticks = args[2];
    ps->hold_ramp = args[3];
    if (ps->amplitude < hold_amp)
        ps->amplitude = hold_amp;
    irq_enable();
}
DECL_COMMAND(command_phase_stepper_set_hold,
             "phase_stepper_set_hold oid=%c amplitude=%hu delay=%u ramp=%u");

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
    uint32_t xfers = stats.xfers, replays = stats.replays;
    uint64_t total_cycles = stats.total_cycles;
    memset(&stats, 0, sizeof(stats));
    irq_enable();
    uint32_t avg_cycles = ticks ? total_cycles / ticks : 0;
    sendf("phase_stepper_state oid=%c phase=%u flags=%c pending=%c ticks=%u"
          " lost_ticks=%u locked_ticks=%u replays=%u xfers=%u"
          " max_cycles=%u avg_cycles=%u tick_clock=%u", oid, phase, flags
          , pending, ticks, lost, locked, replays, xfers, max_cycles
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
    timer_running = 0;
}
DECL_SHUTDOWN(phase_stepper_shutdown);
