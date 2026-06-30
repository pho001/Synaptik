package tuning.measure;

import trace.ExecutionTrace;

/**
 * Measurement output for one candidate workload execution.
 *
 * @param policy policy used for the measurement
 * @param trace compile/prepare/run traces included by the policy
 * @param steadyStateStats aggregate latency statistics in milliseconds
 */
public record MeasurementResult(
        MeasurementPolicy policy,
        ExecutionTrace trace,
        MeasurementStatistics steadyStateStats
) {
    public MeasurementResult {
        policy = policy == null ? MeasurementPolicy.defaults() : policy;
        trace = trace == null ? new ExecutionTrace(null, null, null) : trace;
        steadyStateStats = steadyStateStats == null ? MeasurementStatistics.zero() : steadyStateStats;
    }
}
