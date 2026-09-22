package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import java.util.Objects;

/**
 * Supplies the exact Metal device context used by one NEG partition analysis and finalization.
 *
 * <p>The immutable value retains a backend-private prerequisite only. Analysis performs no native
 * work; finalization uses the same context to compile the persistent executable.</p>
 *
 * @param context non-null open context retained without taking ownership
 */
record MetalNegAnalysisInputs(MetalDeviceContext context) implements BackendAnalysisInputs {
    /**
     * Validates the borrowed context reference.
     *
     * @param context non-null context whose owner must outlive preparation and prepared execution
     * @throws NullPointerException if {@code context} is {@code null}
     */
    MetalNegAnalysisInputs {
        Objects.requireNonNull(context, "context");
    }
}
