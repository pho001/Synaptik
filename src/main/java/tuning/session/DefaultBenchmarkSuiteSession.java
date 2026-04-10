package tuning.session;

import tuning.measure.MeasurementEngine;
import tuning.report.BenchmarkReport;
import tuning.report.BenchmarkSuiteReport;
import tuning.validate.ValidationEngine;
import tuning.workload.WorkloadSpec;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class DefaultBenchmarkSuiteSession implements BenchmarkSuiteSession {
    private final BenchmarkSuiteRequest request;
    private final MeasurementEngine measurementEngine;
    private final ValidationEngine validationEngine;

    DefaultBenchmarkSuiteSession(
            BenchmarkSuiteRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine
    ) {
        this.request = Objects.requireNonNull(request, "request cannot be null");
        this.measurementEngine = Objects.requireNonNull(measurementEngine, "measurementEngine cannot be null");
        this.validationEngine = Objects.requireNonNull(validationEngine, "validationEngine cannot be null");
    }

    @Override
    public BenchmarkSuiteReport run() {
        List<BenchmarkReport> reports = new ArrayList<>(request.workloads().size());
        for (WorkloadSpec workload : request.workloads()) {
            BenchmarkRequest workloadRequest = new BenchmarkRequest(
                    workload,
                    request.entries(),
                    request.measurement(),
                    request.validation(),
                    request.report()
            );
            reports.add(BenchmarkSession.create(workloadRequest, measurementEngine, validationEngine).run());
        }
        return new BenchmarkSuiteReport(OffsetDateTime.now(), reports);
    }
}
