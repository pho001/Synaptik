package io.github.pho001.synaptik.trace.payload;

/** Identifies the trace-owned numerical-profile fact retained by a prepared unit. */
public enum TraceNumericalProfile {
    /** The producer prepared the unit under strict IEEE-oriented semantics. */
    STRICT_IEEE,
    /** The producer prepared the unit under the accelerator result contract. */
    ACCELERATOR
}
