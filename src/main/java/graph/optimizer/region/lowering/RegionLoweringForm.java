package graph.optimizer.region.lowering;

/**
 * Concrete execution form available inside a backend-owned region.
 */
public enum RegionLoweringForm {
    BACKEND_PRIMITIVE,
    BACKEND_DAG,
    MATMUL_EPILOGUE,
    FUSED_ELEMENTWISE,
    LAYOUT_REPAIR,
    REDUCTION_SUBDAG,
    NORMALIZATION_SUBDAG,
    SDPA,
    CONV_POOL,
    LOSS_SUBDAG,
    CPU_FUSED_LOOP,
    UNSUPPORTED
}
