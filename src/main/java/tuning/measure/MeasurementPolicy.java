package tuning.measure;

import graph.execution.PublicationPolicy;

/**
 * Controls how a candidate workload is measured.
 *
 * <p>Measurement is intentionally separate from search and persistence. Engines
 * use this policy to decide which trace phases to report and how many
 * steady-state samples to collect; callers decide how to compare or store the
 * resulting {@link MeasurementResult}.</p>
 *
 * @param warmupIters unreported executions before steady-state sampling
 * @param measureIters executions per measured repeat; must be at least one
 * @param repeats number of timing samples; must be at least one
 * @param measureCompile whether compile trace is included in the result
 * @param measurePrepare whether prepare trace is included in the result
 * @param measureColdRun whether one traced cold run is included
 * @param measureSteadyState whether steady-state latency samples are collected
 * @param captureStepTrace whether run traces should include per-step detail
 * @param publicationPolicy values to publish back to user-visible tensors during measured runs
 */
public record MeasurementPolicy(
        int warmupIters,
        int measureIters,
        int repeats,
        boolean measureCompile,
        boolean measurePrepare,
        boolean measureColdRun,
        boolean measureSteadyState,
        boolean captureStepTrace,
        PublicationPolicy publicationPolicy
) {
    public MeasurementPolicy {
        if (warmupIters < 0) {
            throw new IllegalArgumentException("warmupIters must be >= 0");
        }
        if (measureIters < 1) {
            throw new IllegalArgumentException("measureIters must be >= 1");
        }
        if (repeats < 1) {
            throw new IllegalArgumentException("repeats must be >= 1");
        }
        publicationPolicy = publicationPolicy == null ? PublicationPolicy.defaultExecution() : publicationPolicy;
    }

    /**
     * Creates a measurement policy with default public execution publication semantics.
     */
    public MeasurementPolicy(
            int warmupIters,
            int measureIters,
            int repeats,
            boolean measureCompile,
            boolean measurePrepare,
            boolean measureColdRun,
            boolean measureSteadyState,
            boolean captureStepTrace
    ) {
        this(
                warmupIters,
                measureIters,
                repeats,
                measureCompile,
                measurePrepare,
                measureColdRun,
                measureSteadyState,
                captureStepTrace,
                PublicationPolicy.defaultExecution()
        );
    }

    /**
     * @return balanced default policy with compile, prepare, cold-run, and
     * steady-state measurement enabled
     */
    public static MeasurementPolicy defaults() {
        return new MeasurementPolicy(
                3,
                10,
                3,
                true,
                true,
                true,
                true,
                false,
                PublicationPolicy.defaultExecution()
        );
    }
}
