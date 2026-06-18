package backend.cpu1.kernels.nn.normalization.layernorm;

/**
 * Dense LayerNorm kernels supported by cpu1.
 */
public enum Cpu1LayerNormKernelId {
    LAYER_NORM_F32_ARRAY_DENSE_SCALAR,
    LAYER_NORM_F64_ARRAY_DENSE_SCALAR,
    LAYER_NORM_BF16_ARRAY_DENSE_SCALAR,
    LAYER_NORM_F32_SEGMENT_DENSE_SCALAR,
    LAYER_NORM_F64_SEGMENT_DENSE_SCALAR,
    LAYER_NORM_BF16_SEGMENT_DENSE_SCALAR
}
