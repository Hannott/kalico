// Phase stepping coil transfers by DMA SPI on stm32f4
//
// Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include "command.h" // shutdown
#include "compiler.h" // ARRAY_SIZE
#include "gpio.h" // struct spi_config
#include "internal.h" // DMA_TypeDef
#include "phase_stepper.h" // phase_board_xfer_start
#include "sched.h" // sched_shutdown

struct dma_map {
    SPI_TypeDef *spi;
    DMA_TypeDef *dma;
    DMA_Stream_TypeDef *stream;
    uint32_t clock_bit;
    uint8_t stream_num, channel;
};

// enable_pclock() derives the clock bit from the peripheral address, which
// does not hold for the DMA controllers, so their bits are listed here.
// Each bus has its own stream, so transfers on different buses run at the
// same time.
static const struct dma_map dma_maps[] = {
    { SPI1, DMA2, DMA2_Stream3, RCC_AHB1ENR_DMA2EN, 3, 3 },
    { SPI2, DMA1, DMA1_Stream4, RCC_AHB1ENR_DMA1EN, 4, 0 },
#ifdef SPI3
    { SPI3, DMA1, DMA1_Stream5, RCC_AHB1ENR_DMA1EN, 5, 0 },
#endif
};

static uint8_t configured[ARRAY_SIZE(dma_maps)];

static const struct dma_map *
lookup_map(SPI_TypeDef *spi)
{
    uint32_t i;
    for (i = 0; i < ARRAY_SIZE(dma_maps); i++)
        if (dma_maps[i].spi == spi)
            return &dma_maps[i];
    return NULL;
}

static void
dma_clear_flags(const struct dma_map *m)
{
    static const uint8_t shifts[] = { 0, 6, 16, 22 };
    uint32_t mask = 0x3d << shifts[m->stream_num & 3];
    if (m->stream_num < 4)
        m->dma->LIFCR = mask;
    else
        m->dma->HIFCR = mask;
}

void
phase_board_xfer_setup(struct spi_config *config)
{
    SPI_TypeDef *spi = config->spi;
    const struct dma_map *m = lookup_map(spi);
    if (!m)
        shutdown("Phase stepping not supported on this SPI bus");
    if (configured[m - dma_maps])
        return;
    RCC->AHB1ENR |= m->clock_bit;
    RCC->AHB1ENR;
    DMA_Stream_TypeDef *stream = m->stream;
    stream->CR = 0;
    while (stream->CR & DMA_SxCR_EN)
        ;
    stream->PAR = (uint32_t)&spi->DR;
    stream->FCR = 0;
    uint32_t cr = ((m->channel << DMA_SxCR_CHSEL_Pos) | DMA_SxCR_PL_1
                   | DMA_SxCR_MINC | DMA_SxCR_DIR_0);
    stream->CR = cr;
    if (stream->CR != cr || stream->PAR != (uint32_t)&spi->DR)
        shutdown("Phase stepper DMA not responding");
    configured[m - dma_maps] = 1;
}

void
phase_board_xfer_start(struct spi_config *config, uint8_t *data
                       , uint_fast8_t len)
{
    const struct dma_map *m = lookup_map(config->spi);
    DMA_Stream_TypeDef *stream = m->stream;
    dma_clear_flags(m);
    stream->M0AR = (uint32_t)data;
    stream->NDTR = len;
    stream->CR |= DMA_SxCR_EN;
    m->spi->CR2 |= SPI_CR2_TXDMAEN;
    // Five bytes take microseconds to send, so the stream must still be
    // enabled here; if not, it never started and nothing was sent.
    if (!(stream->CR & DMA_SxCR_EN))
        shutdown("Phase stepper DMA transfer did not start");
}

int
phase_board_xfer_done(struct spi_config *config)
{
    const struct dma_map *m = lookup_map(config->spi);
    SPI_TypeDef *spi = m->spi;
    return (!(m->stream->CR & DMA_SxCR_EN)
            && (spi->SR & (SPI_SR_TXE | SPI_SR_BSY)) == SPI_SR_TXE);
}

void
phase_board_xfer_finish(struct spi_config *config)
{
    SPI_TypeDef *spi = config->spi;
    spi->CR2 &= ~SPI_CR2_TXDMAEN;
    // The transfer was transmit only: drop the received bytes and clear
    // the resulting overrun so the next polled transfer starts clean.
    while (spi->SR & SPI_SR_RXNE)
        (void)spi->DR;
    (void)spi->SR;
}

void
phase_board_xfer_abort(struct spi_config *config)
{
    lookup_map(config->spi)->stream->CR &= ~DMA_SxCR_EN;
    phase_board_xfer_finish(config);
}

uint32_t
phase_board_xfer_bus(struct spi_config *config)
{
    return (uint32_t)config->spi;
}
