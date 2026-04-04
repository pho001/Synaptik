package tuning.measure;

import graph.execution.trace.ExecutionTrace;

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
