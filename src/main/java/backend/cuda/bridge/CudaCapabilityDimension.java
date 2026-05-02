package backend.cuda.bridge;

/**
 * Independent CUDA capability dimensions surfaced in diagnostics and reports.
 */
public enum CudaCapabilityDimension {
    NATIVE_LIBRARY,
    CUDA_RUNTIME,
    CONTEXT,
    GRAPH_EXECUTION_ABI,
    BUFFER_BINDING_ABI,
    LAYOUT_ABI_V2,
    DTYPE_ROLE,
    DAG_PRIMITIVE,
    VENDOR_LIBRARY_ROUTE,
    HARDWARE_DEVICE,
    TOOLCHAIN
}
