package backend.metal.bridge;

/**
 * Runtime path used by a prepared Metal executable for one execution attempt.
 */
public enum MetalMpsBridgeExecutionPath {
    /**
     * The selected Metal region was replayed through CPU fallback steps.
     */
    CPU_FALLBACK,

    /**
     * The current FFM bridge copied Java tensor arrays into native memory, executed MPSGraph,
     * and copied outputs back to Java tensor arrays.
     */
    TENSOR_ARRAY_COPY,

    /**
     * Reserved for future execution through explicit native buffer bindings.
     */
    BUFFER_BINDING
}
