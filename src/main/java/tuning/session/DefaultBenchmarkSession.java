package tuning.session;

import tuning.candidate.Candidate;
import tuning.measure.MeasurementEngine;
import tuning.measure.MeasurementResult;
import tuning.report.BenchmarkCandidateReport;
import tuning.report.BenchmarkReport;
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
        List<BenchmarkCandidateReport> reports = new ArrayList<>(request.candidates().size());
        for (Candidate candidate : request.candidates()) {
            try {
                WorkloadInstance workload = request.workload().instantiate(new WorkloadEnvironment(candidate.profile()));
                ValidationResult validation = validationEngine.validate(candidate, request.workload(), workload, request.validation());
                if (!validation.valid()) {
                    reports.add(BenchmarkCandidateReport.failure(candidate, validation, validation.reason()));
                    continue;
                }
                MeasurementResult measurement = measurementEngine.measure(candidate, workload, request.measurement());
                reports.add(BenchmarkCandidateReport.success(candidate, validation, measurement));
            } catch (Exception ex) {
                reports.add(BenchmarkCandidateReport.failure(
                        candidate,
                        ValidationResult.failure(ex.getMessage()),
                        ex.getClass().getSimpleName() + ": " + ex.getMessage()
                ));
            }
        }
        return BenchmarkReport.of(request.workload().name(), reports);
    }
}
