package graph.compile.planning.region.specialization;

/**
 * Graph-level region specialization families.
 */
public enum RegionSpecializationKind {
    MSE_LOSS,
    SDPA_BACKWARD,
    MATMUL_RELU,
    MATMUL_ADD_BIAS,
    MATMUL_ADD_BIAS_RELU
}
