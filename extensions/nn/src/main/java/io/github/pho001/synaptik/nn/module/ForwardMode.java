package io.github.pho001.synaptik.nn.module;

/**
 * Selects the local forward behavior that a {@link Module} exposes to one concrete layer call.
 *
 * <p>The mode is composition metadata only. It neither evaluates a tensor expression nor creates
 * compiler, runtime, backend, or optimizer state.</p>
 */
public enum ForwardMode {
    /**
     * Directs a concrete layer to construct its training variant of a forward expression.
     *
     * <p>The concrete layer defines that variant; this enum constant neither executes the
     * expression nor provides optimizer behavior.</p>
     */
    TRAINING,

    /**
     * Directs a concrete layer to construct its evaluation variant of a forward expression.
     *
     * <p>The concrete layer defines that variant; this enum constant neither executes the
     * expression nor provides optimizer behavior.</p>
     */
    EVALUATION
}
