package graph.compile.planning.region.specialization;

/**
 * Backend acceptance decision for a graph-level region specialization candidate.
 *
 * @param accepted whether the candidate can become a specialized primitive unit
 * @param reason stable diagnostic reason for traces
 */
public record RegionSpecializationDecision(
        boolean accepted,
        String reason
) {
    public RegionSpecializationDecision {
        reason = reason == null ? "" : reason;
    }

    /**
     * Creates an accepted decision.
     *
     * @param reason acceptance reason
     * @return accepted decision
     */
    public static RegionSpecializationDecision accept(String reason) {
        return new RegionSpecializationDecision(true, reason);
    }

    /**
     * Creates a rejected decision.
     *
     * @param reason rejection reason
     * @return rejected decision
     */
    public static RegionSpecializationDecision reject(String reason) {
        return new RegionSpecializationDecision(false, reason);
    }
}
