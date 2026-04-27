package tuning.benchmark;

import config.profile.ExecutionProfile;
import tuning.measure.MeasurementEngine;
import tuning.measure.MeasurementResult;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkReport;
import tuning.validate.ValidationEngine;
import tuning.validate.ValidationResult;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class DefaultBenchmarkSession implements BenchmarkSession {
    private final BenchmarkRequest request;
    private final MeasurementEngine measurementEngine;
    private final ValidationEngine validationEngine;

    DefaultBenchmarkSession(
            BenchmarkRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine
    ) {
        this.request = Objects.requireNonNull(request, "request cannot be null");
        this.measurementEngine = Objects.requireNonNull(measurementEngine, "measurementEngine cannot be null");
        this.validationEngine = Objects.requireNonNull(validationEngine, "validationEngine cannot be null");
    }

    @Override
    public BenchmarkReport run() {
        List<BenchmarkEntry> entries = request.entries();
        List<BenchmarkCandidateReport> reports = new ArrayList<>(entries.size());
        for (BenchmarkEntry entry : entries) {
            try {
                WorkloadEnvironment environment = new WorkloadEnvironment(entry.profile());
                WorkloadInstance validationWorkload = request.workload().instantiate(environment);
                ValidationResult validation = validationEngine.validate(entry.toCandidate(), request.workload(), validationWorkload, request.validation());
                if (!validation.valid()) {
                    reports.add(BenchmarkCandidateReport.failure(entry, validation, validation.reason()));
                    continue;
                }
                WorkloadInstance measurementWorkload = request.workload().instantiate(environment);
                MeasurementResult measurement = measurementEngine.measure(entry.toCandidate(), measurementWorkload, request.measurement());
                reports.add(BenchmarkCandidateReport.success(entry, validation, measurement));
            } catch (Exception ex) {
                reports.add(BenchmarkCandidateReport.failure(
                        entry,
                        ValidationResult.failure(ex.getMessage()),
                        ex.getClass().getSimpleName() + ": " + ex.getMessage()
                ));
            }
        }
        return BenchmarkReport.of(request.workload().name(), reports);
    }
}
