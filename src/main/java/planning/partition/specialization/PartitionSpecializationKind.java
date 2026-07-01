package planning.partition.specialization;

/**
 * Graph-level partition specialization families.
 */
public enum PartitionSpecializationKind {
    MSE_LOSS,
    SDPA_BACKWARD,
    MATMUL_RELU,
    MATMUL_ADD_BIAS,
    MATMUL_ADD_BIAS_RELU
}
