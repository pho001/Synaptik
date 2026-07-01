package backend.metal.exec;

/**
 * Prepare-time execution route selected inside an already chosen Metal GPU partition.
 */
public enum MetalExecutionRoute {
    MPS_GRAPH,
    CUSTOM_KERNEL,
    TENSOR_ARRAY,
    CPU_FALLBACK,
    UNAVAILABLE_REQUIRED
}
