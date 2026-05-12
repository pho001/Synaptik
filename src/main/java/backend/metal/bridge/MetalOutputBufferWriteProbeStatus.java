package backend.metal.bridge;

/**
 * Result classification for the diagnostic MPSGraph output-buffer write proof.
 */
public enum MetalOutputBufferWriteProbeStatus {
    /**
     * The no-copy output bytes matched a copied reference execution.
     */
    MATCHES_COPIED_RESULT,
    /**
     * The output buffers still contained their pre-run sentinel bytes after the no-copy execution.
     */
    UNCHANGED_SENTINEL,
    /**
     * The no-copy output buffers changed, but did not match the copied reference execution.
     */
    MISMATCHED_RESULT,
    /**
     * The bridge or shim cannot run this proof.
     */
    UNSUPPORTED
}
