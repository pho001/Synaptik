package backend.metal.bridge;

/**
 * Classification of native-side Metal output copy behavior.
 */
public enum MetalNativeCopyStrategy {
    MPSGRAPH_RESULT_COPY,
    TRUE_OUTPUT_BUFFER_WRITE,
    UNKNOWN_OR_UNPROVEN
}
