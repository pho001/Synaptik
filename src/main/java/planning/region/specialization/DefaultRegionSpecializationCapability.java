package planning.region.specialization;

import planning.partition.PartitionTarget;

/**
 * Default backend specialization acceptance hook used by region planning.
 */
public final class DefaultRegionSpecializationCapability implements RegionSpecializationCapability {
    @Override
    public RegionSpecializationDecision evaluate(
            PartitionTarget target,
            RegionSpecializationCandidate candidate
    ) {
        if (candidate == null) {
            return RegionSpecializationDecision.reject("candidate-null");
        }
        if (target == PartitionTarget.CPU) {
            return switch (candidate.kind()) {
                case MSE_LOSS -> RegionSpecializationDecision.accept("cpu1-mse-loss-executable");
                case SDPA_BACKWARD -> RegionSpecializationDecision.accept("cpu1-sdpa-backward-executable");
                case MATMUL_RELU -> RegionSpecializationDecision.accept("cpu1-matmul-relu-executable");
                case MATMUL_ADD_BIAS -> RegionSpecializationDecision.accept("cpu1-matmul-add-bias-executable");
                case MATMUL_ADD_BIAS_RELU -> RegionSpecializationDecision.accept("cpu1-matmul-add-bias-relu-executable");
            };
        }
        return RegionSpecializationDecision.reject("backend-specialization-unsupported:" + safeTarget(target));
    }

    private static String safeTarget(PartitionTarget target) {
        return target == null ? "NONE" : target.name();
    }
}
