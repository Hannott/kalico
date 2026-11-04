// Phase stepping updates from a dedicated timer on stm32
//
// Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include "autoconf.h" // CONFIG_PHASE_STEPPING_FREQ
#include "board/armcm_boot.h" // armcm_enable_irq
#include "internal.h" // enable_pclock
#include "phase_stepper.h" // phase_board_timer_start

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
