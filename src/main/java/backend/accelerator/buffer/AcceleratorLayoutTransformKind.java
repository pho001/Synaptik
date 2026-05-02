package backend.accelerator.buffer;

/**
 * Backend-neutral execution class for a GPU layout transform decision.
 */
public enum AcceleratorLayoutTransformKind {
    METADATA_ONLY_VIEW,
    DENSE_GPU_MATERIALIZATION,
    BROADCAST_GPU_MATERIALIZATION,
    STRIDED_NATIVE_COMPUTE,
    UNSUPPORTED
}
