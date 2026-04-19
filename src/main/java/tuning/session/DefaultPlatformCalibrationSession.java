package tuning.session;

import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.PlatformRuntimeProfile;
import tuning.measure.MeasurementEngine;
import tuning.measure.MeasurementPolicy;
import tuning.measure.MeasurementResult;
import tuning.report.BenchmarkCandidateReport;
import tuning.report.BenchmarkReport;
import tuning.report.BenchmarkSuiteReport;
import tuning.store.HardwareFingerprint;
import tuning.store.PlatformRuntimeProfileStore;
import tuning.validate.ValidationEngine;
import tuning.validate.ValidationResult;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;
import tuning.workload.WorkloadSpec;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class DefaultPlatformCalibrationSession implements PlatformCalibrationSession {
    private final PlatformCalibrationRequest request;
    private final MeasurementEngine measurementEngine;
    private final ValidationEngine validationEngine;
    private final PlatformRuntimeProfileStore profileStore;

    DefaultPlatformCalibrationSession(
            PlatformCalibrationRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine,
            PlatformRuntimeProfileStore profileStore
    ) {
        this.request = Objects.requireNonNull(request, "request cannot be null");
        this.measurementEngine = Objects.requireNonNull(measurementEngine, "measurementEngine cannot be null");
        this.validationEngine = Objects.requireNonNull(validationEngine, "validationEngine cannot be null");
        this.profileStore = Objects.requireNonNull(profileStore, "profileStore cannot be null");
    }

    @Override
    public PlatformCalibrationResult run() {
        PlatformRuntimeProfile current = request.seedRuntimeProfile();
        List<PlatformCalibrationStepResult> results = new ArrayList<>(request.steps().size());
        emit(new PlatformCalibrationProgressEvent(
                PlatformCalibrationProgressPhase.STARTED,
                request.platformId(),
                "",
                0,
                request.steps().size(),
                "",
                0,
                0,
                "",
                0,
                0,
                "",
                Double.NaN,
                "platform calibration started"
        ));

        for (int stepIndex = 0; stepIndex < request.steps().size(); stepIndex++) {
            PlatformCalibrationStep step = request.steps().get(stepIndex);
            emit(new PlatformCalibrationProgressEvent(
                    PlatformCalibrationProgressPhase.FAMILY_STARTED,
                    request.platformId(),
                    step.family().name(),
                    stepIndex + 1,
                    request.steps().size(),
                    "",
                    0,
                    step.workloads().size(),
                    "",
                    0,
                    0,
                    "",
                    Double.NaN,
                    step.name()
            ));
            PlatformRuntimeCandidateSpace candidateSpace = step.candidateSpaceFactory().create(current);
            var generated = candidateSpace.generate(step.workloads().getFirst());
            var entries = generated.stream()
                    .map(candidate -> BenchmarkEntry.candidate(
                            candidate.name(),
                            ExecutionProfileAssembler.assemble(
                                    request.profileName(),
                                    candidate.name(),
                                    request.dataType(),
                                    request.executionMode(),
                                    candidate.runtimeProfile(),
                                    request.graphPolicy()
                            )
                    ))
                    .toList();
            BenchmarkSuiteReport suiteReport = runSuiteWithProgress(step, stepIndex, entries);
            List<PlatformCalibrationCandidateSummary> candidateSummaries = suiteReport.candidateSummaries().stream()
                    .filter(summary -> summary.role() == BenchmarkEntryRole.CANDIDATE)
                    .map(summary -> new PlatformCalibrationCandidateSummary(
                            summary.candidateName(),
                            Map.of("candidateName", summary.candidateName()),
                            step.scorePolicy().score(summary.candidateName(), suiteReport)
                    ))
                    .toList();
            PlatformCalibrationCandidateSummary winner = candidateSummaries.stream()
                    .filter(summary -> summary.score().valid())
                    .min(Comparator.comparingDouble(summary -> summary.score().score()))
                    .orElseGet(() -> candidateSummaries.stream().findFirst().orElseThrow(
                            () -> new IllegalStateException("No candidate summary produced for calibration step " + step.name())
                    ));
            PlatformRuntimeProfile currentProfile = current;
            RuntimeProfileCandidate selectedRuntimeCandidate = generated.stream()
                    .filter(candidate -> candidate.name().equals(winner.candidateId()))
                    .findFirst()
                    .orElseGet(() -> new RuntimeProfileCandidate("seed", currentProfile, Map.of()));
            PlatformRuntimeProfile selectedRuntimeProfile = selectedRuntimeCandidate.runtimeProfile();
            ExecutionProfile selectedExecutionProfile = ExecutionProfileAssembler.assemble(
                    request.profileName(),
                    selectedRuntimeCandidate.name(),
                    request.dataType(),
                    request.executionMode(),
                    selectedRuntimeProfile,
                    request.graphPolicy()
            );
            results.add(new PlatformCalibrationStepResult(
                    step.name(),
                    step.family(),
                    current,
                    suiteReport,
                    candidateSummaries,
                    winner,
                    selectedRuntimeProfile,
                    selectedExecutionProfile,
                    winner.score(),
                    step.scorePolicy().metricName()
            ));
            emit(new PlatformCalibrationProgressEvent(
                    PlatformCalibrationProgressPhase.FAMILY_COMPLETED,
                    request.platformId(),
                    step.family().name(),
                    stepIndex + 1,
                    request.steps().size(),
                    "",
                    step.workloads().size(),
                    step.workloads().size(),
                    winner.candidateId(),
                    generated.size(),
                    generated.size(),
                    winner.candidateId(),
                    winner.score().score(),
                    "family completed"
            ));
            current = selectedRuntimeProfile;
        }

        boolean persisted = false;
        if (request.outputProfilePath() != null) {
            profileStore.save(request.outputProfilePath(), current);
            persisted = true;
        }

        PlatformCalibrationResult result = new PlatformCalibrationResult(
                request.platformId(),
                HardwareFingerprint.capture(),
                request.profileName(),
                request.graphPolicy(),
                request.seedRuntimeProfile(),
                current,
                results,
                request.outputProfilePath(),
                persisted,
                OffsetDateTime.now()
        );
        emit(new PlatformCalibrationProgressEvent(
                PlatformCalibrationProgressPhase.COMPLETED,
                request.platformId(),
                "",
                request.steps().size(),
                request.steps().size(),
                "",
                0,
                0,
                "",
                0,
                0,
                "",
                Double.NaN,
                "platform calibration completed"
        ));
        return result;
    }

    private BenchmarkSuiteReport runSuiteWithProgress(
            PlatformCalibrationStep step,
            int stepIndex,
            List<BenchmarkEntry> entries
    ) {
        List<BenchmarkReport> reports = new ArrayList<>(step.workloads().size());
        for (int workloadIndex = 0; workloadIndex < step.workloads().size(); workloadIndex++) {
            WorkloadSpec workloadSpec = step.workloads().get(workloadIndex);
            emit(new PlatformCalibrationProgressEvent(
                    PlatformCalibrationProgressPhase.WORKLOAD_STARTED,
                    request.platformId(),
                    step.family().name(),
                    stepIndex + 1,
                    request.steps().size(),
                    workloadSpec.name(),
                    workloadIndex + 1,
                    step.workloads().size(),
                    "",
                    0,
                    entries.size(),
                    "",
                    Double.NaN,
                    "workload started"
            ));
            List<BenchmarkCandidateReport> candidateReports = new ArrayList<>(entries.size());
            String leaderId = "";
            double leaderScore = Double.POSITIVE_INFINITY;
            for (int candidateIndex = 0; candidateIndex < entries.size(); candidateIndex++) {
                BenchmarkEntry entry = entries.get(candidateIndex);
                emit(new PlatformCalibrationProgressEvent(
                        PlatformCalibrationProgressPhase.CANDIDATE_VALIDATING,
                        request.platformId(),
                        step.family().name(),
                        stepIndex + 1,
                        request.steps().size(),
                        workloadSpec.name(),
                        workloadIndex + 1,
                        step.workloads().size(),
                        entry.name(),
                        candidateIndex + 1,
                        entries.size(),
                        leaderId,
                        leaderScore,
                        "validating candidate"
                ));
                try {
                    WorkloadEnvironment environment = new WorkloadEnvironment(entry.profile());
                    WorkloadInstance validationWorkload = workloadSpec.instantiate(environment);
                    ValidationResult validation = validationEngine.validate(entry.toCandidate(), workloadSpec, validationWorkload, step.preset().benchmarkValidation());
                    if (!validation.valid()) {
                        candidateReports.add(BenchmarkCandidateReport.failure(entry, validation, validation.reason()));
                        emit(new PlatformCalibrationProgressEvent(
                                PlatformCalibrationProgressPhase.CANDIDATE_INVALID,
                                request.platformId(),
                                step.family().name(),
                                stepIndex + 1,
                                request.steps().size(),
                                workloadSpec.name(),
                                workloadIndex + 1,
                                step.workloads().size(),
                                entry.name(),
                                candidateIndex + 1,
                                entries.size(),
                                leaderId,
                                leaderScore,
                                validation.reason()
                        ));
                        continue;
                    }
                    emit(new PlatformCalibrationProgressEvent(
                            PlatformCalibrationProgressPhase.CANDIDATE_MEASURING,
                            request.platformId(),
                            step.family().name(),
                            stepIndex + 1,
                            request.steps().size(),
                            workloadSpec.name(),
                            workloadIndex + 1,
                            step.workloads().size(),
                            entry.name(),
                            candidateIndex + 1,
                            entries.size(),
                            leaderId,
                            leaderScore,
                            "measuring candidate"
                    ));
                    WorkloadInstance measurementWorkload = workloadSpec.instantiate(environment);
                    MeasurementResult measurement = measurementEngine.measure(entry.toCandidate(), measurementWorkload, calibrationMeasurementPolicy(request, step));
                    candidateReports.add(BenchmarkCandidateReport.success(entry, validation, measurement));
                    double median = measurement.steadyStateStats().medianMs();
                    if (Double.isFinite(median) && median < leaderScore) {
                        leaderId = entry.name();
                        leaderScore = median;
                    }
                    emit(new PlatformCalibrationProgressEvent(
                            PlatformCalibrationProgressPhase.CANDIDATE_MEASURED,
                            request.platformId(),
                            step.family().name(),
                            stepIndex + 1,
                            request.steps().size(),
                            workloadSpec.name(),
                            workloadIndex + 1,
                            step.workloads().size(),
                            entry.name(),
                            candidateIndex + 1,
                            entries.size(),
                            leaderId,
                            leaderScore,
                            "candidate measured"
                    ));
                } catch (Exception ex) {
                    candidateReports.add(BenchmarkCandidateReport.failure(
                            entry,
                            ValidationResult.failure(ex.getMessage()),
                            ex.getClass().getSimpleName() + ": " + ex.getMessage()
                    ));
                    emit(new PlatformCalibrationProgressEvent(
                            PlatformCalibrationProgressPhase.CANDIDATE_FAILED,
                            request.platformId(),
                            step.family().name(),
                            stepIndex + 1,
                            request.steps().size(),
                            workloadSpec.name(),
                            workloadIndex + 1,
                            step.workloads().size(),
                            entry.name(),
                            candidateIndex + 1,
                            entries.size(),
                            leaderId,
                            leaderScore,
                            ex.getClass().getSimpleName() + ": " + ex.getMessage()
                    ));
                }
            }
            reports.add(BenchmarkReport.of(workloadSpec.name(), candidateReports));
        }
        BenchmarkSuiteReport suiteReport = new BenchmarkSuiteReport(OffsetDateTime.now(), reports);
        for (PlatformCalibrationCandidateSummary summary : suiteReport.candidateSummaries().stream()
                .filter(s -> s.role() == BenchmarkEntryRole.CANDIDATE)
                .map(s -> new PlatformCalibrationCandidateSummary(
                        s.candidateName(),
                        Map.of("candidateName", s.candidateName()),
                        step.scorePolicy().score(s.candidateName(), suiteReport)
                ))
                .toList()) {
            emit(new PlatformCalibrationProgressEvent(
                    PlatformCalibrationProgressPhase.CANDIDATE_SCORED,
                    request.platformId(),
                    step.family().name(),
                    stepIndex + 1,
                    request.steps().size(),
                    "",
                    step.workloads().size(),
                    step.workloads().size(),
                    summary.candidateId(),
                    0,
                    entries.size(),
                    summary.candidateId(),
                    summary.score().score(),
                    summary.score().explanation()
            ));
        }
        return suiteReport;
    }

    private void emit(PlatformCalibrationProgressEvent event) {
        request.progressListener().onEvent(event);
    }

    private static MeasurementPolicy calibrationMeasurementPolicy(
            PlatformCalibrationRequest request,
            PlatformCalibrationStep step
    ) {
        MeasurementPolicy override = request.measurement();
        return override == null ? step.preset().benchmarkMeasurement() : override;
    }
}
