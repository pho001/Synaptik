package planning.region.specialization;

import planning.partition.PartitionTarget;

/**
 * Backend hook for accepting graph-level region specialization candidates.
 */
@FunctionalInterface
public interface RegionSpecializationCapability {
    /**
     * Evaluates whether a backend target accepts a specialization candidate.
     *
     * @param target partition target being optimized
     * @param candidate specialization candidate
     * @return acceptance decision
     */
    RegionSpecializationDecision evaluate(
            PartitionTarget target,
            RegionSpecializationCandidate candidate
    );

    /**
     * Returns a capability that rejects every specialization.
     *
     * @return rejecting capability
     */
    static RegionSpecializationCapability rejecting() {
        return (target, candidate) -> RegionSpecializationDecision.reject("backend-specialization-default-reject");
    }
}
