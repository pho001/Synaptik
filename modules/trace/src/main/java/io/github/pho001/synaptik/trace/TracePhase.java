package io.github.pho001.synaptik.trace;

/**
 * Identifies the lifecycle phase in which a diagnostic fact occurs.
 *
 * <p>A backend diagnostic uses the phase during which the backend fact occurs; a backend is a
 * producer or payload family, not a separate lifecycle phase.</p>
 */
public enum TracePhase {
    /**
     * Classifies capture, validation, transformation, ownership, partitioning, logical-memory, or
     * publication-planning diagnostics.
     */
    COMPILE,

    /**
     * Classifies backend preparation, route selection, prepared partitions or units, prepared
     * memory, or prepared-schedule diagnostics.
     */
    PREPARE,

    /** Classifies invocation, execution, transfer, materialization, step, or publication diagnostics. */
    RUN
}
