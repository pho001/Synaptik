package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import java.util.Objects;
import java.util.Optional;

/**
 * Supplies the exact Metal device context used by one NEG partition analysis and finalization.
 *
 * <p>The immutable value retains a backend-private prerequisite only. Analysis performs no native
 * work; finalization uses the same context to compile the persistent executable.</p>
 *
 * @param context non-null open context retained without taking ownership
 * @param tuningHandoff non-null optional Metal-owned handoff; a present selection is treated as
 *     untrusted input and authenticated against freshly generated analysis facts
 */
record MetalNegAnalysisInputs(
        MetalDeviceContext context,
        Optional<BackendPartitionTuningHandoff<
                MetalNegTuningBatch, MetalNegTuningDecision>> tuningHandoff)
        implements BackendAnalysisInputs {
    /**
     * Creates the ordinary no-decision input used by existing preparation.
     *
     * @param context non-null borrowed context
     * @throws NullPointerException if {@code context} is {@code null}
     */
    MetalNegAnalysisInputs(MetalDeviceContext context) {
        this(context, Optional.empty());
    }

    /**
     * Validates the borrowed context reference.
     *
     * @param context non-null context whose owner must outlive preparation and prepared execution
     * @param tuningHandoff non-null optional untrusted backend-local tuning handoff
     * @throws NullPointerException if a component is {@code null}
     */
    MetalNegAnalysisInputs {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(tuningHandoff, "tuningHandoff");
    }
}
