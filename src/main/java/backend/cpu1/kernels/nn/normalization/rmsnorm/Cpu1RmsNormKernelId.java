package backend.cpu1.kernels.nn.normalization.rmsnorm;

/**
 * Dense RMSNorm kernels supported by cpu1.
 */
public enum Cpu1RmsNormKernelId {
    RMS_NORM_F32_ARRAY_DENSE_SCALAR,
    RMS_NORM_F64_ARRAY_DENSE_SCALAR,
    RMS_NORM_BF16_ARRAY_DENSE_SCALAR,
    RMS_NORM_F32_SEGMENT_DENSE_SCALAR,
    RMS_NORM_F64_SEGMENT_DENSE_SCALAR,
    RMS_NORM_BF16_SEGMENT_DENSE_SCALAR
}
