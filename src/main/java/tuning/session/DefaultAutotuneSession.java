package tuning.session;

import config.profile.ExecutionProfile;
import tuning.candidate.Candidate;
import tuning.measure.MeasurementEngine;
import tuning.measure.MeasurementResult;
import tuning.report.BenchmarkCandidateReport;
import tuning.report.TuningSummary;
import tuning.search.SearchContext;
import tuning.search.SearchResult;
import tuning.search.SearchStrategy;
import tuning.store.BestProfileRecord;
import tuning.store.BestProfileStore;
import tuning.store.HardwareFingerprint;
import tuning.store.TuningHistoryEntry;
import tuning.store.TuningHistoryStore;
import tuning.store.WorkloadFingerprint;
import tuning.validate.ValidationEngine;
import tuning.validate.ValidationResult;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

final class DefaultAutotuneSession implements AutotuneSession {
    private final AutotuneRequest request;
    private final SearchStrategy searchStrategy;
    private final MeasurementEngine measurementEngine;
    private final ValidationEngine validationEngine;
    private final BestProfileStore bestProfileStore;
    private final TuningHistoryStore historyStore;

    DefaultAutotuneSession(
            AutotuneRequest request,
            SearchStrategy searchStrategy,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine,
            BestProfileStore bestProfileStore,
            TuningHistoryStore historyStore
    ) {
        this.request = Objects.requireNonNull(request, "request cannot be null");
        this.searchStrategy = Objects.requireNonNull(searchStrategy, "searchStrategy cannot be null");
        this.measurementEngine = Objects.requireNonNull(measurementEngine, "measurementEngine cannot be null");
        this.validationEngine = Objects.requireNonNull(validationEngine, "validationEngine cannot be null");
        this.bestProfileStore = Objects.requireNonNull(bestProfileStore, "bestProfileStore cannot be null");
        this.historyStore = Objects.requireNonNull(historyStore, "historyStore cannot be null");
    }

    @Override
    public TuningResult run() {
        List<BenchmarkCandidateReport> evaluated = new ArrayList<>();
        SearchContext context = new SearchContext(request, request.candidateSpace());
        int totalCandidateCount = request.candidateSpace().generate(request.workload()).size();
        ProgressState progress = new ProgressState(totalCandidateCount);
        emit(progress.event(
                AutotuneProgressPhase.STARTED,
                0,
                0,
                "",
                "",
                "autotune started"
        ));
        SearchResult initial = searchStrategy.search(context);
        Set<String> seenFingerprints = new HashSet<>();
        emit(progress.event(
                AutotuneProgressPhase.SEARCH_BATCH,
                0,
                initial.selectedCandidates().size(),
                "",
                "",
                "initial search batch selected"
        ));

        evaluateCandidates(initial.selectedCandidates(), evaluated, seenFingerprints, progress, 0);
        emit(progress.event(
                AutotuneProgressPhase.ROUND_COMPLETED,
                0,
                initial.selectedCandidates().size(),
                "",
                "",
                "initial batch completed"
        ));

        if (searchStrategy.supportsRefinement()) {
            for (int round = 1; round < request.search().maxRounds(); round++) {
                SearchResult refined = searchStrategy.refine(context, evaluated, round, seenFingerprints);
                if (refined.selectedCandidates().isEmpty()) {
                    break;
                }
                emit(progress.event(
                        AutotuneProgressPhase.SEARCH_BATCH,
                        round,
                        refined.selectedCandidates().size(),
                        "",
                        "",
                        "refinement batch selected"
                ));
                evaluateCandidates(refined.selectedCandidates(), evaluated, seenFingerprints, progress, round);
                emit(progress.event(
                        AutotuneProgressPhase.ROUND_COMPLETED,
                        round,
                        refined.selectedCandidates().size(),
                        "",
                        "",
                        "refinement batch completed"
                ));
            }
        }

        List<BenchmarkCandidateReport> finalists = evaluated.stream()
                .filter(BenchmarkCandidateReport::success)
                .filter(report -> report.measurement() != null)
                .sorted(Comparator.comparingDouble(report -> report.measurement().steadyStateStats().medianMs()))
                .limit(Math.max(1, request.search().beamWidth()))
                .toList();

        ExecutionProfile bestProfile = finalists.isEmpty() ? null : finalists.getFirst().candidate().profile();
        String summary = "evaluated=" + evaluated.size()
                + ", valid=" + finalists.size()
                + ", selected=" + seenFingerprints.size();
        boolean persisted = persist(bestProfile, evaluated, finalists, summary);
        int historyEntries = request.persistence().persistHistory() && request.persistence().historyPath() != null ? evaluated.size() : 0;
        double bestMedian = finalists.isEmpty() ? Double.POSITIVE_INFINITY : finalists.getFirst().measurement().steadyStateStats().medianMs();
        TuningResult result = new TuningResult(
                bestProfile,
                finalists,
                summary,
                new TuningSummary(
                        searchStrategy.getClass().getSimpleName(),
                        seenFingerprints.size(),
                        evaluated.size(),
                        (int) evaluated.stream().filter(BenchmarkCandidateReport::success).count(),
                        finalists.size(),
                        historyEntries,
                        bestMedian
                ),
                persisted
        );
        emit(progress.event(
                AutotuneProgressPhase.COMPLETED,
                request.search().maxRounds(),
                finalists.size(),
                bestProfile == null ? "" : bestProfile.candidateName(),
                "",
                "autotune completed"
        ));
        return result;
    }

    private void evaluateCandidates(
            List<Candidate> candidates,
            List<BenchmarkCandidateReport> evaluated,
            Set<String> seenFingerprints,
            ProgressState progress,
            int round
    ) {
        for (Candidate candidate : candidates) {
            String fp = tuning.candidate.CandidateFingerprint.of(candidate);
            if (!seenFingerprints.add(fp)) {
                continue;
            }
            emit(progress.event(
                    AutotuneProgressPhase.CANDIDATE_VALIDATING,
                    round,
                    candidates.size(),
                    candidate.name(),
                    fp,
                    "validating candidate"
            ));
            try {
                WorkloadInstance workload = request.workload().instantiate(new WorkloadEnvironment(candidate.profile()));
                ValidationResult validation = validationEngine.validate(candidate, request.workload(), workload, request.validation());
                if (!validation.valid()) {
                    BenchmarkCandidateReport report = BenchmarkCandidateReport.failure(candidate, validation, validation.reason());
                    evaluated.add(report);
                    progress.accept(report);
                    emit(progress.event(
                            AutotuneProgressPhase.CANDIDATE_INVALID,
                            round,
                            candidates.size(),
                            candidate.name(),
                            fp,
                            validation.reason()
                    ));
                    continue;
                }
                emit(progress.event(
                        AutotuneProgressPhase.CANDIDATE_MEASURING,
                        round,
                        candidates.size(),
                        candidate.name(),
                        fp,
                        "measuring candidate"
                ));
                MeasurementResult measurement = measurementEngine.measure(candidate, workload, request.measurement());
                BenchmarkCandidateReport report = BenchmarkCandidateReport.success(candidate, validation, measurement);
                evaluated.add(report);
                progress.accept(report);
                emit(progress.event(
                        AutotuneProgressPhase.CANDIDATE_MEASURED,
                        round,
                        candidates.size(),
                        candidate.name(),
                        fp,
                        "candidate measured"
                ));
            } catch (Exception ex) {
                BenchmarkCandidateReport report = BenchmarkCandidateReport.failure(
                        candidate,
                        ValidationResult.failure(ex.getMessage()),
                        ex.getClass().getSimpleName() + ": " + ex.getMessage()
                );
                evaluated.add(report);
                progress.accept(report);
                emit(progress.event(
                        AutotuneProgressPhase.CANDIDATE_FAILED,
                        round,
                        candidates.size(),
                        candidate.name(),
                        fp,
                        report.failureReason()
                ));
            }
        }
    }

    private void emit(AutotuneProgressEvent event) {
        request.progressListener().onEvent(event);
    }

    private boolean persist(
            ExecutionProfile bestProfile,
            List<BenchmarkCandidateReport> evaluated,
            List<BenchmarkCandidateReport> finalists,
            String summary
    ) {
        var policy = request.persistence();
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        WorkloadFingerprint workload = WorkloadFingerprint.of(
                request.workload(),
                finalists.isEmpty() ? tuning.workload.WorkloadMetadata.of(request.workload().name(), request.workload().kind())
                        : request.workload().instantiate(new WorkloadEnvironment(finalists.getFirst().candidate().profile())).metadata(),
                bestProfile
        );

        if (policy.persistHistory() && policy.historyPath() != null) {
            for (BenchmarkCandidateReport report : evaluated) {
                double median = report.measurement() == null ? Double.POSITIVE_INFINITY : report.measurement().steadyStateStats().medianMs();
                double mean = report.measurement() == null ? Double.POSITIVE_INFINITY : report.measurement().steadyStateStats().meanMs();
                historyStore.append(policy.historyPath(), new TuningHistoryEntry(
                        tuning.candidate.CandidateFingerprint.of(report.candidate()),
                        report.candidate().name(),
                        report.success(),
                        median,
                        mean,
                        median,
                        report.failureReason(),
                        summary,
                        java.time.OffsetDateTime.now(),
                        hardware,
                        workload
                ));
            }
        }

        if (policy.persistBestProfile() && policy.bestProfilePath() != null && bestProfile != null) {
            double score = finalists.isEmpty() ? Double.POSITIVE_INFINITY : finalists.getFirst().measurement().steadyStateStats().medianMs();
            bestProfileStore.save(policy.bestProfilePath(), new BestProfileRecord(
                    hardware,
                    workload,
                    bestProfile,
                    score,
                    java.time.OffsetDateTime.now()
            ));
            return true;
        }
        return false;
    }

    private static final class ProgressState {
        private final int totalCandidateCount;
        private int evaluatedCount;
        private int validCount;
        private String bestCandidateName = "";
        private double bestMedianMs = Double.POSITIVE_INFINITY;

        private ProgressState(int totalCandidateCount) {
            this.totalCandidateCount = Math.max(0, totalCandidateCount);
        }

        private void accept(BenchmarkCandidateReport report) {
            evaluatedCount++;
            if (report.success() && report.measurement() != null) {
                validCount++;
                double median = report.measurement().steadyStateStats().medianMs();
                if (!Double.isFinite(bestMedianMs) || median < bestMedianMs) {
                    bestMedianMs = median;
                    bestCandidateName = report.candidate().name();
                }
            }
        }

        private AutotuneProgressEvent event(
                AutotuneProgressPhase phase,
                int round,
                int selectedCount,
                String candidateName,
                String candidateFingerprint,
                String message
        ) {
            return new AutotuneProgressEvent(
                    phase,
                    round,
                    totalCandidateCount,
                    selectedCount,
                    evaluatedCount,
                    validCount,
                    candidateName,
                    candidateFingerprint,
                    bestCandidateName,
                    bestMedianMs,
                    message
            );
        }
    }
}
