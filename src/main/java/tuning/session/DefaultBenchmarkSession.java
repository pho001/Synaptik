package tuning.session;

import config.profile.ExecutionProfile;
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
        List<Candidate> candidates = withBaselines(request.candidates());
        List<BenchmarkCandidateReport> reports = new ArrayList<>(candidates.size());
        for (Candidate candidate : candidates) {
            try {
                WorkloadInstance workload = request.workload().instantiate(new WorkloadEnvironment(candidate.profile()));
                ValidationResult validation = validationEngine.validate(candidate, request.workload(), workload, request.validation());
                if (!validation.valid()) {
                    reports.add(BenchmarkCandidateReport.failure(candidate, validation, validation.reason(), baselineKind(candidate)));
                    continue;
                }
                MeasurementResult measurement = measurementEngine.measure(candidate, workload, request.measurement());
                reports.add(BenchmarkCandidateReport.success(candidate, validation, measurement, baselineKind(candidate)));
            } catch (Exception ex) {
                reports.add(BenchmarkCandidateReport.failure(
                        candidate,
                        ValidationResult.failure(ex.getMessage()),
                        ex.getClass().getSimpleName() + ": " + ex.getMessage(),
                        baselineKind(candidate)
                ));
            }
        }
        return BenchmarkReport.of(request.workload().name(), reports);
    }

    private List<Candidate> withBaselines(List<Candidate> original) {
        if (original == null || original.isEmpty()) {
            return List.of();
        }
        ArrayList<Candidate> out = new ArrayList<>(original);
        boolean hasNoOpt = original.stream().anyMatch(c -> "BASELINE_NO_OPT".equals(c.name()));
        boolean hasNoOptConservative = original.stream().anyMatch(c -> "BASELINE_NO_OPT_CONSERVATIVE_RUNTIME".equals(c.name()));
        ExecutionProfile reference = original.getFirst().profile();
        if (request.baselines().includeNoOptBaseline() && !hasNoOpt) {
            out.add(new Candidate("BASELINE_NO_OPT", BenchmarkBaselineProfiles.noOptimization(reference)));
        }
        if (request.baselines().includeNoOptConservativeRuntimeBaseline() && !hasNoOptConservative) {
            out.add(new Candidate(
                    "BASELINE_NO_OPT_CONSERVATIVE_RUNTIME",
                    BenchmarkBaselineProfiles.noOptimizationConservativeRuntime(reference)
            ));
        }
        return List.copyOf(out);
    }

    private static tuning.report.BenchmarkBaselineKind baselineKind(Candidate candidate) {
        if (candidate == null) {
            return tuning.report.BenchmarkBaselineKind.NONE;
        }
        return switch (candidate.name()) {
            case "BASELINE_NO_OPT" -> tuning.report.BenchmarkBaselineKind.NO_OPT;
            case "BASELINE_NO_OPT_CONSERVATIVE_RUNTIME" -> tuning.report.BenchmarkBaselineKind.NO_OPT_CONSERVATIVE_RUNTIME;
            default -> tuning.report.BenchmarkBaselineKind.NONE;
        };
    }
}
