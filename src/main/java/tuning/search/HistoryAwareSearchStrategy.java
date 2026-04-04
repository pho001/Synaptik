package tuning.search;

import config.profile.ExecutionProfile;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateFingerprint;
import tuning.candidate.CandidateSpace;
import tuning.candidate.ListCandidateSpace;
import tuning.store.BestProfileResolver;
import tuning.store.HardwareFingerprint;
import tuning.store.TuningHistoryEntry;
import tuning.store.TuningHistoryStore;
import tuning.store.WorkloadFingerprint;
import tuning.workload.WorkloadMetadata;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class HistoryAwareSearchStrategy implements SearchStrategy {
    private final SearchStrategy delegate;
    private final BestProfileResolver bestProfileResolver;
    private final TuningHistoryStore historyStore;
    private final Path bestProfilePath;
    private final Path historyPath;

    public HistoryAwareSearchStrategy(
            SearchStrategy delegate,
            BestProfileResolver bestProfileResolver,
            TuningHistoryStore historyStore,
            Path bestProfilePath,
            Path historyPath
    ) {
        this.delegate = Objects.requireNonNull(delegate, "delegate cannot be null");
        this.bestProfileResolver = Objects.requireNonNull(bestProfileResolver, "bestProfileResolver cannot be null");
        this.historyStore = Objects.requireNonNull(historyStore, "historyStore cannot be null");
        this.bestProfilePath = bestProfilePath;
        this.historyPath = historyPath;
    }

    @Override
    public SearchResult search(SearchContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        CandidateSpace original = context.candidateSpace();
        List<Candidate> generated = original.generate(context.request().workload());
        if (generated.isEmpty()) {
            return delegate.search(context);
        }

        HardwareFingerprint hardware = HardwareFingerprint.capture();
        WorkloadFingerprint workload = WorkloadFingerprint.of(
                context.request().workload(),
                WorkloadMetadata.of(context.request().workload().name(), context.request().workload().kind()),
                generated.getFirst().profile()
        );
        Map<String, Candidate> byFingerprint = new LinkedHashMap<>();
        for (Candidate candidate : generated) {
            byFingerprint.putIfAbsent(CandidateFingerprint.of(candidate), candidate);
        }

        List<Candidate> ordered = new ArrayList<>();

        bestProfileResolver.resolve(bestProfilePath, hardware, workload).ifPresent(profile -> {
            String fp = CandidateFingerprint.of(profile);
            Candidate candidate = byFingerprint.get(fp);
            if (candidate != null) {
                ordered.add(candidate);
            } else {
                ordered.add(new Candidate(profile.candidateName(), profile));
            }
        });

        List<TuningHistoryEntry> history = historyPath == null ? List.of() : historyStore.loadAll(historyPath).stream()
                .filter(entry -> entry.hardware().key().equals(hardware.key()))
                .filter(entry -> entry.workload().key().equals(workload.key()))
                .sorted(Comparator.comparingDouble(TuningHistoryEntry::score))
                .toList();

        for (TuningHistoryEntry entry : history) {
            Candidate candidate = byFingerprint.get(entry.fingerprint());
            if (candidate == null) {
                continue;
            }
            if (!entry.valid() && context.request().search().allowPruning()) {
                continue;
            }
            if (ordered.stream().anyMatch(existing -> CandidateFingerprint.of(existing).equals(entry.fingerprint()))) {
                continue;
            }
            ordered.add(candidate);
        }

        for (Candidate candidate : generated) {
            String fp = CandidateFingerprint.of(candidate);
            if (ordered.stream().anyMatch(existing -> CandidateFingerprint.of(existing).equals(fp))) {
                continue;
            }
            boolean invalidHistory = history.stream().anyMatch(entry -> entry.fingerprint().equals(fp) && !entry.valid());
            if (invalidHistory && context.request().search().allowPruning()) {
                continue;
            }
            ordered.add(candidate);
        }

        SearchContext reordered = new SearchContext(
                context.request(),
                new ListCandidateSpace(List.copyOf(ordered))
        );
        return delegate.search(reordered);
    }

    @Override
    public boolean supportsRefinement() {
        return delegate.supportsRefinement();
    }

    @Override
    public SearchResult refine(
            SearchContext context,
            List<tuning.report.BenchmarkCandidateReport> evaluatedSoFar,
            int round,
            Set<String> seenFingerprints
    ) {
        return delegate.refine(context, evaluatedSoFar, round, seenFingerprints);
    }
}
