package planning.partition.specialization;

import planning.partition.PartitionTarget;

/**
 * Backend hook for accepting graph-level partition specialization candidates.
 */
@FunctionalInterface
public interface PartitionSpecializationCapability {
    /**
     * Evaluates whether a backend target accepts a specialization candidate.
     *
     * @param target partition target being optimized
     * @param candidate specialization candidate
     * @return acceptance decision
     */
    PartitionSpecializationDecision evaluate(
            PartitionTarget target,
            PartitionSpecializationCandidate candidate
    );

    /**
     * Returns a capability that rejects every specialization.
     *
     * @return rejecting capability
     */
    static PartitionSpecializationCapability rejecting() {
        return (target, candidate) -> PartitionSpecializationDecision.reject("backend-specialization-default-reject");
    }
}
