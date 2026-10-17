package io.github.pho001.synaptik.training;

/** Selects how one successful forward/backward run uses its published parameter gradients. */
public enum GradientMode {
    /**
     * Ignores previously accumulated gradients, performs one optimizer update from this run, and
     * leaves the accumulator empty. An unsuccessful run preserves the earlier accumulator.
     */
    RESET_AND_STEP,

    /**
     * Adds this run's gradients to the pending accumulator without changing parameter values,
     * optimizer slots, or optimizer step number.
     */
    ACCUMULATE,

    /**
     * Adds this run's gradients to the pending accumulator, performs one optimizer update from the
     * complete sum, and leaves the accumulator empty.
     */
    ACCUMULATE_AND_STEP
}
