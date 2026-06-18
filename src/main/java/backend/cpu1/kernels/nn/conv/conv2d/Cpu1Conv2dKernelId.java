package backend.cpu1.kernels.nn.conv.conv2d;

/**
 * Dense direct CONV2D kernels supported by cpu1.
 */
public enum Cpu1Conv2dKernelId {
    CONV2D_F32_ARRAY_DENSE_SCALAR,
    CONV2D_F64_ARRAY_DENSE_SCALAR,
    CONV2D_BF16_ARRAY_DENSE_SCALAR,
    CONV2D_F32_SEGMENT_DENSE_SCALAR,
    CONV2D_F64_SEGMENT_DENSE_SCALAR,
    CONV2D_BF16_SEGMENT_DENSE_SCALAR
}
