package tuning.session;

import config.profile.ExecutionProfile;
import tuning.candidate.Candidate;
import tuning.measure.MeasurementEngine;
import tuning.measure.MeasurementResult;
import tuning.report.BenchmarkCandidateReport;
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
        SearchResult initial = searchStrategy.search(context);
        Set<String> seenFingerprints = new HashSet<>();

        evaluateCandidates(initial.selectedCandidates(), evaluated, seenFingerprints);

        if (searchStrategy.supportsRefinement()) {
            for (int round = 1; round < request.search().maxRounds(); round++) {
                SearchResult refined = searchStrategy.refine(context, evaluated, round, seenFingerprints);
                if (refined.selectedCandidates().isEmpty()) {
                    break;
                }
                evaluateCandidates(refined.selectedCandidates(), evaluated, seenFingerprints);
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
        boolean persisted = persist(bestProfile, finalists, summary);
        return new TuningResult(bestProfile, finalists, summary, persisted);
    }

    private void evaluateCandidates(
            List<Candidate> candidates,
            List<BenchmarkCandidateReport> evaluated,
            Set<String> seenFingerprints
    ) {
        for (Candidate candidate : candidates) {
            String fp = tuning.candidate.CandidateFingerprint.of(candidate);
            if (!seenFingerprints.add(fp)) {
                continue;
            }
            try {
                WorkloadInstance workload = request.workload().instantiate(new WorkloadEnvironment(candidate.profile()));
                ValidationResult validation = validationEngine.validate(candidate, request.workload(), workload, request.validation());
                if (!validation.valid()) {
                    evaluated.add(BenchmarkCandidateReport.failure(candidate, validation, validation.reason()));
                    continue;
                }
                MeasurementResult measurement = measurementEngine.measure(candidate, workload, request.measurement());
                evaluated.add(BenchmarkCandidateReport.success(candidate, validation, measurement));
            } catch (Exception ex) {
                evaluated.add(BenchmarkCandidateReport.failure(
                        candidate,
                        ValidationResult.failure(ex.getMessage()),
                        ex.getClass().getSimpleName() + ": " + ex.getMessage()
                ));
            }
        }
    }

    private boolean persist(
            ExecutionProfile bestProfile,
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
            for (BenchmarkCandidateReport finalist : finalists) {
                double median = finalist.measurement() == null ? Double.POSITIVE_INFINITY : finalist.measurement().steadyStateStats().medianMs();
                double mean = finalist.measurement() == null ? Double.POSITIVE_INFINITY : finalist.measurement().steadyStateStats().meanMs();
                historyStore.append(policy.historyPath(), new TuningHistoryEntry(
                        finalist.candidate().name(),
                        finalist.success(),
                        median,
                        mean,
                        median,
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
}
