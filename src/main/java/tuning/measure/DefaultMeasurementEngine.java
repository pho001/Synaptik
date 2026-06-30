package tuning.measure;

import graph.CompiledGraph;
import runtime.execution.PreparedExecution;
import trace.ExecutionTrace;
import trace.compile.CompileTrace;
import trace.prepare.PrepareTrace;
import trace.execution.RunTrace;
import tuning.candidate.Candidate;
import tuning.workload.WorkloadInstance;
import training.optimizer.AdamOptimizer;
import training.optimizer.SgdOptimizer;
import training.optimizer.TrainingOptimizer;

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
                compileModeFor(candidate.profile().mode(), policy.executionMode()),
                workload.backendIntentPlan()
        );

        PreparedExecution prepared = compiled.prepare(candidate.profile().runtime());
        TrainingOptimizer optimizer = optimizerFor(policy.executionMode());
        try {
            RunTrace reportRunTrace = RunTrace.empty(candidate.profile().mode());
            boolean needsTrace = policy.measureColdRun() || policy.captureStepTrace();

            if (needsTrace && !policy.measureSteadyState()) {
                reportRunTrace = executeTraced(prepared, candidate, policy, optimizer);
            }

            MeasurementStatistics stats = MeasurementStatistics.zero();
            if (policy.measureSteadyState()) {
                for (int i = 0; i < policy.warmupIters(); i++) {
                    execute(prepared, candidate, policy, optimizer);
                }
                if (needsTrace) {
                    reportRunTrace = executeTraced(prepared, candidate, policy, optimizer);
                }
                double[] samples = new double[policy.repeats()];
                for (int r = 0; r < policy.repeats(); r++) {
                    long start = System.nanoTime();
                    for (int i = 0; i < policy.measureIters(); i++) {
                        execute(prepared, candidate, policy, optimizer);
                    }
                    long end = System.nanoTime();
                    samples[r] = (end - start) / 1_000_000.0d / policy.measureIters();
                }
                stats = summarize(samples);
            }

            ExecutionTrace trace = new ExecutionTrace(
                    policy.measureCompile() ? compiled.compileTrace() : CompileTrace.skipped(),
                    policy.measurePrepare() ? prepared.prepareTrace() : PrepareTrace.skipped(),
                    reportRunTrace
            );
            return new MeasurementResult(policy, trace, stats);
        } finally {
            if (optimizer != null) {
                optimizer.close();
            }
            prepared.close();
        }
    }

    private static void execute(
            PreparedExecution prepared,
            Candidate candidate,
            MeasurementPolicy policy,
            TrainingOptimizer optimizer
    ) {
        if (!policy.executionMode().optimizerStep()) {
            prepared.execute(candidate.profile().mode(), policy.publicationPolicy());
            return;
        }
        requireTrainingProfile(candidate, policy);
        prepared.executeOptimizerStep(optimizer, policy.publicationPolicy());
    }

    private static RunTrace executeTraced(
            PreparedExecution prepared,
            Candidate candidate,
            MeasurementPolicy policy,
            TrainingOptimizer optimizer
    ) {
        if (!policy.executionMode().optimizerStep()) {
            return prepared.executeTraced(candidate.profile().mode(), policy.publicationPolicy());
        }
        requireTrainingProfile(candidate, policy);
        return prepared.executeOptimizerStepTraced(optimizer, policy.publicationPolicy());
    }

    private static TrainingOptimizer optimizerFor(MeasurementExecutionMode mode) {
        return switch (mode) {
            case GRAPH_EXECUTION -> null;
            case OPTIMIZER_STEP_SGD -> new SgdOptimizer(0.01f);
            case OPTIMIZER_STEP_ADAM -> new AdamOptimizer(0.01f);
        };
    }

    private static void requireTrainingProfile(Candidate candidate, MeasurementPolicy policy) {
        if (candidate.profile().mode() != runtime.contract.ExecutionMode.FORWARD_BACKWARD) {
            throw new IllegalArgumentException(policy.executionMode()
                    + " measurement requires an execution profile with FORWARD_BACKWARD mode.");
        }
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

    private static tensor.CompileMode compileModeFor(
            runtime.contract.ExecutionMode mode,
            MeasurementExecutionMode measurementMode
    ) {
        return measurementMode.optimizerStep() || mode == runtime.contract.ExecutionMode.FORWARD_BACKWARD
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
