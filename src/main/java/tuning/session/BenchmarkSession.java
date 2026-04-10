package tuning.session;

import tuning.measure.DefaultMeasurementEngine;
import tuning.measure.MeasurementEngine;
import tuning.report.BenchmarkReport;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.ValidationEngine;

public interface BenchmarkSession {
    BenchmarkReport run();

    static BenchmarkSession create(BenchmarkRequest request) {
        return create(request, new DefaultMeasurementEngine(), new DefaultValidationEngine());
    }

    static BenchmarkSession create(
            BenchmarkRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine
    ) {
        return new DefaultBenchmarkSession(request, measurementEngine, validationEngine);
    }

}
