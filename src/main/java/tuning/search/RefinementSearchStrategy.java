package tuning.search;

import tuning.candidate.Candidate;
import tuning.candidate.CandidateFingerprint;
import tuning.candidate.RefinableCandidateSpace;
import tuning.report.BenchmarkCandidateReport;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

public final class RefinementSearchStrategy implements SearchStrategy {
    private final SearchStrategy seedStrategy;
    private final int beamWidth;
    private final int maxNeighborsPerSeed;

    public RefinementSearchStrategy(SearchStrategy seedStrategy, int beamWidth, int maxNeighborsPerSeed) {
        this.seedStrategy = Objects.requireNonNull(seedStrategy, "seedStrategy cannot be null");
        this.beamWidth = Math.max(1, beamWidth);
        this.maxNeighborsPerSeed = Math.max(1, maxNeighborsPerSeed);
    }

    @Override
    public SearchResult search(SearchContext context) {
        return seedStrategy.search(context);
    }

    @Override
    public boolean supportsRefinement() {
        return true;
    }

    @Override
    public SearchResult refine(
            SearchContext context,
            List<BenchmarkCandidateReport> evaluatedSoFar,
            int round,
            Set<String> seenFingerprints
    ) {
        Objects.requireNonNull(context, "context cannot be null");
        if (!(context.candidateSpace() instanceof RefinableCandidateSpace refinable)) {
            return new SearchResult(List.of(), null);
        }
        if (evaluatedSoFar == null || evaluatedSoFar.isEmpty()) {
            return new SearchResult(List.of(), null);
        }

        List<BenchmarkCandidateReport> seeds = evaluatedSoFar.stream()
                .filter(BenchmarkCandidateReport::success)
                .filter(report -> report.measurement() != null)
                .sorted(Comparator.comparingDouble(report -> report.measurement().steadyStateStats().medianMs()))
                .limit(beamWidth)
                .toList();
        if (seeds.isEmpty()) {
            return new SearchResult(List.of(), null);
        }

        LinkedHashMap<String, Candidate> next = new LinkedHashMap<>();
        for (BenchmarkCandidateReport seed : seeds) {
            List<Candidate> neighbors = refinable.neighbors(seed.candidate(), context.request().workload());
            int accepted = 0;
            for (Candidate neighbor : neighbors) {
                String fp = CandidateFingerprint.of(neighbor);
                if (seenFingerprints.contains(fp) || next.containsKey(fp)) {
                    continue;
                }
                next.put(fp, neighbor);
                accepted++;
                if (accepted >= maxNeighborsPerSeed || next.size() >= context.request().search().maxCandidates()) {
                    break;
                }
            }
            if (next.size() >= context.request().search().maxCandidates()) {
                break;
            }
        }

        List<Candidate> selected = new ArrayList<>(next.values());
        Candidate preferred = selected.isEmpty() ? null : selected.getFirst();
        return new SearchResult(List.copyOf(selected), preferred);
    }
}
