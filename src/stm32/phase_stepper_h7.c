// Phase stepping coil transfers through the SPI FIFO on stm32h7
//
// Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include "board/io.h" // writeb
#include "command.h" // shutdown
#include "gpio.h" // struct spi_config
#include "internal.h" // SPI_TypeDef
#include "phase_stepper.h" // phase_board_xfer_start
#include "sched.h" // sched_shutdown

// The SPI FIFOs hold at least 8 bytes, enough for a whole coil register
// write, so the hardware sends it on its own without DMA. (DMA would also
// need a buffer outside the ram used here, as the DMA controllers cannot
// reach DTCM.)
#define FIFO_SIZE 8

static SPI_TypeDef *bus_spi;

void
phase_board_xfer_setup(struct spi_config *config)
{
    if (bus_spi && bus_spi != config->spi)
        shutdown("Phase steppers must share one SPI bus");
    bus_spi = config->spi;
}

void
phase_board_xfer_start(struct spi_config *config, uint8_t *data
                       , uint_fast8_t len)
{
    if (len > FIFO_SIZE)
        shutdown("Phase stepper transfer too long");
    SPI_TypeDef *spi = bus_spi;
    spi->CR2 = len << SPI_CR2_TSIZE_Pos;
    // Enable SPI and start transfer, these MUST be set in this sequence
    spi->CR1 = SPI_CR1_SSI | SPI_CR1_SPE;
    spi->CR1 = SPI_CR1_SSI | SPI_CR1_CSTART | SPI_CR1_SPE;
    while (len--)
        writeb((void *)&spi->TXDR, *data++);
}

int
phase_board_xfer_done(void)
{
    return bus_spi->SR & SPI_SR_EOT;
}

void
phase_board_xfer_finish(void)
{
    SPI_TypeDef *spi = bus_spi;
    // The transfer was transmit only: drop the received bytes
    while (spi->SR & (SPI_SR_RXWNE | SPI_SR_RXPLVL))
        (void)readb((void *)&spi->RXDR);
    spi->IFCR = 0xFFFFFFFF;
    spi->CR1 = SPI_CR1_SSI;
}

void
phase_board_xfer_abort(void)
{
    SPI_TypeDef *spi = bus_spi;
    // Disabling the SPI stops the transfer and flushes its FIFOs
    spi->IFCR = 0xFFFFFFFF;
    spi->CR1 = SPI_CR1_SSI;
}
