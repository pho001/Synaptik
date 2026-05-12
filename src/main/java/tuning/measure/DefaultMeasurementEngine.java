package tuning.measure;

import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.trace.ExecutionTrace;
import graph.execution.trace.RunTrace;
import tuning.candidate.Candidate;
import tuning.workload.WorkloadInstance;

import java.util.Arrays;

/**
 * Default measurement engine backed by {@link CompiledGraph}.
 *
 * <p>The engine is stateless and safe to share between sessions. Each
 * {@link #measure(Candidate, WorkloadInstance, MeasurementPolicy)} call compiles
 * the workload root with the candidate optimizer, prepares execution with the
 * candidate runtime profile, optionally records traces, and samples steady-state
 * execution time with {@link System#nanoTime()}.</p>
 */
public final class DefaultMeasurementEngine implements MeasurementEngine {
    /**
     * Measures one candidate/workload pair.
     *
     * @param candidate candidate execution profile
     * @param workload instantiated workload graph
     * @param policy measurement controls
     * @return traces and steady-state latency statistics
     * @throws IllegalArgumentException if any argument is {@code null}
     */
    @Override
    public MeasurementResult measure(Candidate candidate, WorkloadInstance workload, MeasurementPolicy policy) {
        if (candidate == null) {
            throw new IllegalArgumentException("candidate cannot be null");
        }
        if (workload == null) {
            throw new IllegalArgumentException("workload cannot be null");
        }
        if (policy == null) {
            throw new IllegalArgumentException("policy cannot be null");
        }

        CompiledGraph compiled = CompiledGraph.compile(
                workload.root(),
                candidate.profile().compile(),
                compileModeFor(candidate.profile().mode())
        );

        PreparedExecution prepared = compiled.prepare(candidate.profile().runtime());

        RunTrace reportRunTrace = RunTrace.empty(candidate.profile().mode());
        boolean needsTrace = policy.measureColdRun() || policy.captureStepTrace();

        if (needsTrace && !policy.measureSteadyState()) {
            reportRunTrace = prepared.executeTraced(candidate.profile().mode(), policy.publicationPolicy());
        }

        MeasurementStatistics stats = MeasurementStatistics.zero();
        if (policy.measureSteadyState()) {
            for (int i = 0; i < policy.warmupIters(); i++) {
                prepared.execute(candidate.profile().mode(), policy.publicationPolicy());
            }
            if (needsTrace) {
                reportRunTrace = prepared.executeTraced(candidate.profile().mode(), policy.publicationPolicy());
            }
            double[] samples = new double[policy.repeats()];
            for (int r = 0; r < policy.repeats(); r++) {
                long start = System.nanoTime();
                for (int i = 0; i < policy.measureIters(); i++) {
                    prepared.execute(candidate.profile().mode(), policy.publicationPolicy());
                }
                long end = System.nanoTime();
                samples[r] = (end - start) / 1_000_000.0d / policy.measureIters();
            }
            stats = summarize(samples);
        }

        ExecutionTrace trace = new ExecutionTrace(
                policy.measureCompile() ? compiled.compileTrace() : graph.execution.trace.CompileTrace.skipped(),
                policy.measurePrepare() ? prepared.prepareTrace() : graph.execution.trace.PrepareTrace.skipped(),
                reportRunTrace
        );
        return new MeasurementResult(policy, trace, stats);
    }

    private static MeasurementStatistics summarize(double[] samplesMs) {
        if (samplesMs == null || samplesMs.length == 0) {
            return MeasurementStatistics.zero();
        }
        double[] sorted = samplesMs.clone();
        Arrays.sort(sorted);
        double mean = Arrays.stream(sorted).average().orElse(0.0d);
        double median = percentile(sorted, 50);
        double p90 = percentile(sorted, 90);
        return new MeasurementStatistics(mean, median, p90);
    }

    private static tensor.CompileMode compileModeFor(backend.runtime.ExecutionMode mode) {
        return mode == backend.runtime.ExecutionMode.FORWARD_BACKWARD
                ? tensor.CompileMode.TRAINING
                : tensor.CompileMode.INFERENCE_ONLY;
    }

    private static double percentile(double[] sortedValues, int p) {
        if (sortedValues.length == 1) {
            return sortedValues[0];
        }
        double rank = (p / 100.0d) * (sortedValues.length - 1);
        int low = (int) Math.floor(rank);
        int high = (int) Math.ceil(rank);
        if (low == high) {
            return sortedValues[low];
        }
        double w = rank - low;
        return sortedValues[low] * (1.0d - w) + sortedValues[high] * w;
    }
}
