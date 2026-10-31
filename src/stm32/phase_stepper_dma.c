// Phase stepping updates from a dedicated timer with DMA SPI on stm32f4
//
// Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include "autoconf.h" // CONFIG_PHASE_STEPPING_FREQ
#include "board/armcm_boot.h" // armcm_enable_irq
#include "command.h" // shutdown
#include "compiler.h" // ARRAY_SIZE
#include "gpio.h" // struct spi_config
#include "internal.h" // enable_pclock
#include "phase_stepper.h" // phase_board_timer_start
#include "sched.h" // sched_shutdown

struct dma_map {
    SPI_TypeDef *spi;
    DMA_TypeDef *dma;
    DMA_Stream_TypeDef *stream;
    uint8_t stream_num, channel;
};

static const struct dma_map dma_maps[] = {
    { SPI1, DMA2, DMA2_Stream3, 3, 3 },
    { SPI2, DMA1, DMA1_Stream4, 4, 0 },
#ifdef SPI3
    { SPI3, DMA1, DMA1_Stream5, 5, 0 },
#endif
};

static const struct dma_map *bus_dma;

static void
dma_clear_flags(void)
{
    static const uint8_t shifts[] = { 0, 6, 16, 22 };
    uint32_t mask = 0x3d << shifts[bus_dma->stream_num & 3];
    if (bus_dma->stream_num < 4)
        bus_dma->dma->LIFCR = mask;
    else
        bus_dma->dma->HIFCR = mask;
}

void
phase_board_xfer_setup(struct spi_config *config)
{
    SPI_TypeDef *spi = config->spi;
    if (bus_dma) {
        if (bus_dma->spi != spi)
            shutdown("Phase steppers must share one SPI bus");
        return;
    }
    const struct dma_map *m = NULL;
    uint32_t i;
    for (i = 0; i < ARRAY_SIZE(dma_maps); i++)
        if (dma_maps[i].spi == spi)
            m = &dma_maps[i];
    if (!m)
        shutdown("Phase stepping not supported on this SPI bus");
    enable_pclock((uint32_t)m->dma);
    DMA_Stream_TypeDef *stream = m->stream;
    stream->CR = 0;
    while (stream->CR & DMA_SxCR_EN)
        ;
    stream->PAR = (uint32_t)&spi->DR;
    stream->FCR = 0;
    stream->CR = ((m->channel << DMA_SxCR_CHSEL_Pos) | DMA_SxCR_PL_1
                  | DMA_SxCR_MINC | DMA_SxCR_DIR_0);
    bus_dma = m;
}

void
phase_board_xfer_start(struct spi_config *config, uint8_t *data
                       , uint_fast8_t len)
{
    DMA_Stream_TypeDef *stream = bus_dma->stream;
    dma_clear_flags();
    stream->M0AR = (uint32_t)data;
    stream->NDTR = len;
    stream->CR |= DMA_SxCR_EN;
    bus_dma->spi->CR2 |= SPI_CR2_TXDMAEN;
}

int
phase_board_xfer_done(void)
{
    SPI_TypeDef *spi = bus_dma->spi;
    return (!(bus_dma->stream->CR & DMA_SxCR_EN)
            && (spi->SR & (SPI_SR_TXE | SPI_SR_BSY)) == SPI_SR_TXE);
}

void
phase_board_xfer_finish(void)
{
    SPI_TypeDef *spi = bus_dma->spi;
    spi->CR2 &= ~SPI_CR2_TXDMAEN;
    // The transfer was transmit only: drop the received bytes and clear
    // the resulting overrun so the next polled transfer starts clean.
    while (spi->SR & SPI_SR_RXNE)
        (void)spi->DR;
    (void)spi->SR;
}

void
phase_board_xfer_abort(void)
{
    bus_dma->stream->CR &= ~DMA_SxCR_EN;
    phase_board_xfer_finish();
}

void
phase_stepper_irq(void)
{
    TIM7->SR = 0;
    phase_stepper_update();
}

void
phase_board_timer_start(void)
{
    enable_pclock(TIM7_BASE);
    TIM7->CR1 = 0;
    TIM7->DIER = 0;
    TIM7->PSC = 0;
    // APB1 runs below the cpu clock, so its timers run at twice its rate
    TIM7->ARR = (2 * get_pclock_frequency(TIM7_BASE)
                 / CONFIG_PHASE_STEPPING_FREQ - 1);
    TIM7->EGR = TIM_EGR_UG;
    TIM7->SR = 0;
    TIM7->DIER = TIM_DIER_UIE;
    TIM7->CR1 = TIM_CR1_CEN;
    // Above the stepper timer, so coil updates stay on the tick grid
    armcm_enable_irq(phase_stepper_irq, TIM7_IRQn, 0);
}

void
phase_board_timer_stop(void)
{
    NVIC_DisableIRQ(TIM7_IRQn);
    TIM7->CR1 = 0;
    TIM7->DIER = 0;
    TIM7->SR = 0;
    NVIC_ClearPendingIRQ(TIM7_IRQn);
}
