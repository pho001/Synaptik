package backend.accelerator.lowering;

/**
 * Backend-neutral operation families used by the GPU lowering coverage matrix.
 */
public enum GpuLoweringOperationFamily {
    MATMUL_LINEAR,
    ELEMENTWISE_CHAIN,
    LAYOUT_VIEW_ADJACENT,
    SOFTMAX_LIKE,
    REDUCTION,
    NORMALIZATION,
    LOSS_ADJACENT,
    ATTENTION,
    CONV_POOL,
    INDEX_SCATTER_GATHER,
    COMPARE_BOOL,
    DTYPE_CONVERSION,
    BACKWARD_ADJACENT
}
