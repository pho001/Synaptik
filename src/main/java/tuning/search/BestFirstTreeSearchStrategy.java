package tuning.search;

import tuning.candidate.Candidate;
import tuning.candidate.CandidateFingerprint;
import tuning.candidate.RefinableCandidateSpace;
import tuning.report.BenchmarkCandidateReport;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class BestFirstTreeSearchStrategy extends AbstractTreeSearchStrategy {
    private final CandidateScoreModel scoreModel;
    private final int maxNeighborsPerNode;

    public BestFirstTreeSearchStrategy(SearchStrategy seedStrategy, CandidateScoreModel scoreModel, int maxNeighborsPerNode) {
        super(seedStrategy);
        this.scoreModel = Objects.requireNonNull(scoreModel, "scoreModel cannot be null");
        this.maxNeighborsPerNode = Math.max(1, maxNeighborsPerNode);
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
        if (frontierFingerprints.isEmpty() || evaluatedSoFar == null || evaluatedSoFar.isEmpty()) {
            return new SearchResult(List.of(), null);
        }

        Map<String, BenchmarkCandidateReport> reportsByFp = new LinkedHashMap<>();
        for (BenchmarkCandidateReport report : evaluatedSoFar) {
            reportsByFp.put(CandidateFingerprint.of(report.candidate()), report);
        }

        BenchmarkCandidateReport bestFrontier = frontierFingerprints.stream()
                .map(reportsByFp::get)
                .filter(Objects::nonNull)
                .filter(BenchmarkCandidateReport::success)
                .filter(report -> report.measurement() != null)
                .min(Comparator.comparingDouble(scoreModel::score))
                .orElse(null);
        if (bestFrontier == null) {
            frontierFingerprints = List.of();
            return new SearchResult(List.of(), null);
        }

        Candidate seed = bestFrontier.candidate();
        String parentFp = CandidateFingerprint.of(seed);
        LinkedHashMap<String, Candidate> next = new LinkedHashMap<>();
        List<String> nextFrontier = new ArrayList<>();
        int accepted = 0;
        for (Candidate neighbor : refinable.neighbors(seed, context.request().workload())) {
            String fp = CandidateFingerprint.of(neighbor);
            if (seenFingerprints.contains(fp) || next.containsKey(fp)) {
                continue;
            }
            next.put(fp, neighbor);
            nextFrontier.add(fp);
            registerChild(neighbor, parentFp, round);
            accepted++;
            if (accepted >= maxNeighborsPerNode || next.size() >= context.request().search().maxCandidates()) {
                break;
            }
        }

        frontierFingerprints = List.copyOf(nextFrontier);
        List<Candidate> selected = new ArrayList<>(next.values());
        return new SearchResult(List.copyOf(selected), selected.isEmpty() ? null : selected.getFirst());
    }
}
