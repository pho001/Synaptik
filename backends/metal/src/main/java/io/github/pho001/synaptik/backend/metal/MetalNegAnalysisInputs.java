package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import java.util.Objects;
import java.util.Optional;

/**
 * Supplies the exact Metal device context used by one supported-operation partition
 * analysis/finalization.
 *
 * <p>The immutable value retains a backend-private prerequisite only. Analysis performs no native
 * work; finalization uses the same context to compile the persistent executable.</p>
 *
 * @param context non-null open context retained without taking ownership
 * @param tuningHandoff non-null optional Metal-owned handoff; a present selection is treated as
 *     untrusted input and authenticated against freshly generated analysis facts
 * @param traceProducer nullable integration-owned producer; null is the allocation-free no-trace
 *     path
 */
record MetalNegAnalysisInputs(
        MetalDeviceContext context,
        Optional<BackendPartitionTuningHandoff<
                MetalNegTuningBatch, MetalNegTuningDecision>> tuningHandoff,
        MetalTraceProducer traceProducer)
        implements BackendAnalysisInputs {
    /**
     * Creates the ordinary no-decision input used by existing preparation.
     *
     * @param context non-null borrowed context
     * @throws NullPointerException if {@code context} is {@code null}
     */
    MetalNegAnalysisInputs(MetalDeviceContext context) {
        this(context, Optional.empty(), null);
    }

    /**
     * Creates the ordinary untraced input with an optional tuning handoff.
     *
     * @param context non-null borrowed context
     * @param tuningHandoff non-null optional untrusted backend-local tuning handoff
     */
    MetalNegAnalysisInputs(
            MetalDeviceContext context,
            Optional<BackendPartitionTuningHandoff<
                    MetalNegTuningBatch, MetalNegTuningDecision>> tuningHandoff) {
        this(context, tuningHandoff, null);
    }

    /**
     * Creates traced no-decision analysis inputs.
     *
     * @param context non-null borrowed context
     * @param traceProducer nullable retained producer; null selects the no-trace path
     */
    MetalNegAnalysisInputs(
            MetalDeviceContext context, MetalTraceProducer traceProducer) {
        this(context, Optional.empty(), traceProducer);
    }

    /**
     * Validates the borrowed context reference.
     *
     * @param context non-null context whose owner must outlive preparation and prepared execution
     * @param tuningHandoff non-null optional untrusted backend-local tuning handoff
     * @param traceProducer nullable integration-owned producer
     * @throws NullPointerException if {@code context} or {@code tuningHandoff} is null
     */
    MetalNegAnalysisInputs {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(tuningHandoff, "tuningHandoff");
    }
}
