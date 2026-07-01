package planning.partition.specialization;

/**
 * Backend acceptance decision for a graph-level partition specialization candidate.
 *
 * @param accepted whether the candidate can become a specialized primitive unit
 * @param reason stable diagnostic reason for traces
 */
public record PartitionSpecializationDecision(
        boolean accepted,
        String reason
) {
    public PartitionSpecializationDecision {
        reason = reason == null ? "" : reason;
    }

    /**
     * Creates an accepted decision.
     *
     * @param reason acceptance reason
     * @return accepted decision
     */
    public static PartitionSpecializationDecision accept(String reason) {
        return new PartitionSpecializationDecision(true, reason);
    }

    /**
     * Creates a rejected decision.
     *
     * @param reason rejection reason
     * @return rejected decision
     */
    public static PartitionSpecializationDecision reject(String reason) {
        return new PartitionSpecializationDecision(false, reason);
    }
}
