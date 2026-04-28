package tuning.benchmark;

import tuning.measure.DefaultMeasurementEngine;
import tuning.measure.MeasurementEngine;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.ValidationEngine;

/**
 * Runs a {@link BenchmarkSuiteRequest} by applying a benchmark request to each
 * workload and aggregating the per-workload reports.
 */
public interface BenchmarkSuiteSession {
    /**
     * Executes the suite.
     *
     * @return aggregate suite report
     */
    BenchmarkSuiteReport run();

    /**
     * Creates a suite session with default engines.
     *
     * @param request suite request
     * @return new suite session
     */
    static BenchmarkSuiteSession create(BenchmarkSuiteRequest request) {
        return create(request, new DefaultMeasurementEngine(), new DefaultValidationEngine());
    }

    /**
     * Creates a suite session with caller-supplied engines.
     *
     * @param request suite request
     * @param measurementEngine engine shared by all workload runs
     * @param validationEngine validation engine shared by all workload runs
     * @return new suite session
     */
    static BenchmarkSuiteSession create(
            BenchmarkSuiteRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine
    ) {
        return new DefaultBenchmarkSuiteSession(request, measurementEngine, validationEngine);
    }
}
