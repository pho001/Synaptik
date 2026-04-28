package tuning.benchmark;

import tuning.measure.DefaultMeasurementEngine;
import tuning.measure.MeasurementEngine;
import tuning.benchmark.report.BenchmarkReport;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.ValidationEngine;

/**
 * Runs a {@link BenchmarkRequest} by validating and measuring caller-supplied
 * profiles.
 *
 * <p>Sessions are intended to be used once. The default implementation catches
 * per-entry validation and measurement failures and records them in the
 * resulting report, allowing remaining entries to continue.</p>
 */
public interface BenchmarkSession {
    /**
     * Executes the benchmark request.
     *
     * @return benchmark report with one candidate report per entry
     */
    BenchmarkReport run();

    /**
     * Creates a benchmark session with default measurement and validation engines.
     *
     * @param request non-null benchmark request
     * @return new session
     */
    static BenchmarkSession create(BenchmarkRequest request) {
        return create(request, new DefaultMeasurementEngine(), new DefaultValidationEngine());
    }

    /**
     * Creates a benchmark session with caller-supplied engines.
     *
     * @param request benchmark request
     * @param measurementEngine engine used to measure valid entries
     * @param validationEngine engine used before measurement
     * @return new session
     */
    static BenchmarkSession create(
            BenchmarkRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine
    ) {
        return new DefaultBenchmarkSession(request, measurementEngine, validationEngine);
    }

}
