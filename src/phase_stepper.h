#ifndef __PHASE_STEPPER_H
#define __PHASE_STEPPER_H

#include <stdint.h> // uint8_t

// Hardware SPI users call these around each transfer so they do not
// collide with the phase stepper's coil current updates on a shared bus.
struct spidev_s;
void phase_stepper_spi_acquire(struct spidev_s *spi, uint8_t receive_data
                               , uint8_t data_len, uint8_t *data);
void phase_stepper_spi_release(void);

// Called by the board timer CONFIG_PHASE_STEPPING_FREQ times a second
void phase_stepper_update(void);

// Board support. A transfer is started with the chip select already
// asserted and must not block longer than one update period.
struct spi_config;
void phase_board_timer_start(void);
void phase_board_timer_stop(void);
void phase_board_xfer_setup(struct spi_config *config);
void phase_board_xfer_start(struct spi_config *config, uint8_t *data
                            , uint_fast8_t len);
int phase_board_xfer_done(void);
void phase_board_xfer_finish(void);
void phase_board_xfer_abort(void);

#endif // phase_stepper.h
