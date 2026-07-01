package trace.compile;

import java.util.List;

/**
 * Partition planning diagnostics captured during compilation.
 *
 * @param totalConsidered number of candidate starts considered
 * @param acceptedCount number of accepted partitions
 * @param rejectedCount number of rejected candidates
 * @param jobs per planning job aggregate diagnostics
 * @param decisions detailed partition decisions
 */
public record PartitionCompileTrace(
        int totalConsidered,
        int acceptedCount,
        int rejectedCount,
        List<JobTrace> jobs,
        List<PartitionDecisionTrace> decisions
) {
    public PartitionCompileTrace {
        totalConsidered = Math.max(0, totalConsidered);
        acceptedCount = Math.max(0, acceptedCount);
        rejectedCount = Math.max(0, rejectedCount);
        jobs = List.copyOf(jobs == null ? List.of() : jobs);
        decisions = List.copyOf(decisions == null ? List.of() : decisions);
    }

    public record JobTrace(
            String strategy,
            String target,
            int totalConsidered,
            int acceptedCount,
            int rejectedCount
    ) {
        public JobTrace {
            strategy = strategy == null ? "GREEDY_MAX_PARTITION" : strategy;
            target = target == null ? "NONE" : target;
            totalConsidered = Math.max(0, totalConsidered);
            acceptedCount = Math.max(0, acceptedCount);
            rejectedCount = Math.max(0, rejectedCount);
        }
    }

    /**
     * Returns an empty partition compile trace.
     *
     * @return empty trace
     */
    public static PartitionCompileTrace empty() {
        return new PartitionCompileTrace(
                0,
                0,
                0,
                List.of(),
                List.of()
        );
    }

    public static PartitionCompileTrace singleJob(
            String strategy,
            String target,
            int totalConsidered,
            int acceptedCount,
            int rejectedCount,
            List<PartitionDecisionTrace> decisions
    ) {
        return new PartitionCompileTrace(
                totalConsidered,
                acceptedCount,
                rejectedCount,
                List.of(new JobTrace(strategy, target, totalConsidered, acceptedCount, rejectedCount)),
                decisions
        );
    }

    public static PartitionCompileTrace forJob(
            String strategy,
            String target,
            List<PartitionDecisionTrace> decisions
    ) {
        List<PartitionDecisionTrace> safeDecisions = List.copyOf(decisions == null ? List.of() : decisions);
        int accepted = (int) safeDecisions.stream().filter(PartitionDecisionTrace::accepted).count();
        return singleJob(
                strategy,
                target,
                safeDecisions.size(),
                accepted,
                safeDecisions.size() - accepted,
                safeDecisions
        );
    }
}
