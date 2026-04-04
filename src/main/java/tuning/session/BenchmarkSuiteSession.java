package tuning.session;

import tuning.measure.DefaultMeasurementEngine;
import tuning.measure.MeasurementEngine;
import tuning.report.BenchmarkSuiteReport;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.ValidationEngine;

public interface BenchmarkSuiteSession {
    BenchmarkSuiteReport run();

    static BenchmarkSuiteSession create(BenchmarkSuiteRequest request) {
        return create(request, new DefaultMeasurementEngine(), new DefaultValidationEngine());
    }

    static BenchmarkSuiteSession create(
            BenchmarkSuiteRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine
    ) {
        return new DefaultBenchmarkSuiteSession(request, measurementEngine, validationEngine);
    }
}
