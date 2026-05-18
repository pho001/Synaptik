package backend.cpu.nativecpu.layout;

/**
 * CPU-native layout classes used by region planning before selecting a physical segment kernel.
 */
public enum NativeCpuLayoutClass {
    DENSE_CONTIGUOUS,
    OFFSET_CONTIGUOUS,
    BROADCAST_READ_DENSE_WRITE,
    LAST_DIM_BIAS_BROADCAST,
    TRANSPOSE_2D_READ_DENSE_WRITE,
    SAME_SHAPE_STRIDED,
    GENERAL_STRIDED_READ_STRIDED_WRITE,
    VIEW_ALIAS_ONLY,
    GENERAL_STRIDED_READ_DENSE_WRITE,
    UNSUPPORTED_LAYOUT
}
