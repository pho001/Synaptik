package backend.cpu.nativecpu;

/**
 * Layout/access capabilities tracked separately from operation existence.
 */
public enum NativeCpuLayoutCapability {
    DENSE,
    OFFSET_CONTIGUOUS,
    SAME_SHAPE_STRIDED_READ,
    ZERO_STRIDE_BROADCAST_READ,
    LAST_DIM_BIAS_BROADCAST,
    TRANSPOSE_PERMUTE_READ_VIEW,
    SELECT_SLICE_OFFSET_VIEW,
    STRIDED_WRITE,
    DENSE_MATERIALIZATION
}
