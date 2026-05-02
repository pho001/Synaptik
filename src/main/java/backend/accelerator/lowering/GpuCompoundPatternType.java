package backend.accelerator.lowering;

/**
 * Stable names for compound GPU region patterns.
 */
public enum GpuCompoundPatternType {
    NONE,
    LINEAR_BIAS_ACTIVATION,
    ELEMENTWISE_CHAIN,
    NORMALIZATION,
    REDUCTION_ADJACENT,
    CPU_FUSED_UNSUPPORTED
}
