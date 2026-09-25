package io.github.pho001.synaptik.trace.payload;

/** Classifies whether a traced preparation or invocation completed successfully. */
public enum TraceOutcomeStatus {
    /** The observed operation completed successfully. */
    SUCCEEDED,
    /** The observed operation failed. */
    FAILED
}
