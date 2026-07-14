package io.github.pho001.synaptik.trace;

/**
 * Classifies the detail or severity of a diagnostic event.
 *
 * <p>The constants progress from the most detailed diagnostics to the most severe. This
 * classification defines no filtering threshold, sink behavior, logging integration, failure
 * handling, or process-exit policy.</p>
 */
public enum TraceLevel {
    /** Classifies the event as the most detailed diagnostic information. */
    TRACE,

    /** Classifies the event as diagnostic information intended for debugging. */
    DEBUG,

    /** Classifies the event as informational. */
    INFO,

    /** Classifies the event as a warning without defining warning handling. */
    WARN,

    /** Classifies the event as an error without implying failure-handling behavior. */
    ERROR
}
