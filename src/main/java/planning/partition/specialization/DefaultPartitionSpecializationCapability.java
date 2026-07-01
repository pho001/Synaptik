package planning.partition.specialization;

import planning.partition.PartitionTarget;

/**
 * Default backend specialization acceptance hook used by partition planning.
 */
public final class DefaultPartitionSpecializationCapability implements PartitionSpecializationCapability {
    @Override
    public PartitionSpecializationDecision evaluate(
            PartitionTarget target,
            PartitionSpecializationCandidate candidate
    ) {
        if (candidate == null) {
            return PartitionSpecializationDecision.reject("candidate-null");
        }
        if (target == PartitionTarget.CPU) {
            return switch (candidate.kind()) {
                case MSE_LOSS -> PartitionSpecializationDecision.accept("cpu1-mse-loss-executable");
                case SDPA_BACKWARD -> PartitionSpecializationDecision.accept("cpu1-sdpa-backward-executable");
                case MATMUL_RELU -> PartitionSpecializationDecision.accept("cpu1-matmul-relu-executable");
                case MATMUL_ADD_BIAS -> PartitionSpecializationDecision.accept("cpu1-matmul-add-bias-executable");
                case MATMUL_ADD_BIAS_RELU -> PartitionSpecializationDecision.accept("cpu1-matmul-add-bias-relu-executable");
            };
        }
        return PartitionSpecializationDecision.reject("backend-specialization-unsupported:" + safeTarget(target));
    }

    private static String safeTarget(PartitionTarget target) {
        return target == null ? "NONE" : target.name();
    }
}
