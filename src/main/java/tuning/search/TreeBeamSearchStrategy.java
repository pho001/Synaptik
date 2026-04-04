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

public final class TreeBeamSearchStrategy implements SearchStrategy {
    private final SearchStrategy seedStrategy;
    private final int beamWidth;
    private final int maxNeighborsPerNode;
    private final Map<String, SearchTreeNode> nodesByFingerprint = new LinkedHashMap<>();
    private List<String> frontierFingerprints = List.of();

    public TreeBeamSearchStrategy(SearchStrategy seedStrategy, int beamWidth, int maxNeighborsPerNode) {
        this.seedStrategy = Objects.requireNonNull(seedStrategy, "seedStrategy cannot be null");
        this.beamWidth = Math.max(1, beamWidth);
        this.maxNeighborsPerNode = Math.max(1, maxNeighborsPerNode);
    }

    @Override
    public SearchResult search(SearchContext context) {
        SearchResult seed = seedStrategy.search(context);
        nodesByFingerprint.clear();
        List<String> frontier = new ArrayList<>(seed.selectedCandidates().size());
        for (Candidate candidate : seed.selectedCandidates()) {
            String fp = CandidateFingerprint.of(candidate);
            frontier.add(fp);
            nodesByFingerprint.put(fp, new SearchTreeNode(
                    fp,
                    candidate.name(),
                    null,
                    0,
                    0
            ));
        }
        frontierFingerprints = List.copyOf(frontier);
        return seed;
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
        if (frontierFingerprints.isEmpty()) {
            return new SearchResult(List.of(), null);
        }

        Map<String, BenchmarkCandidateReport> reportsByFp = new LinkedHashMap<>();
        for (BenchmarkCandidateReport report : evaluatedSoFar) {
            reportsByFp.put(CandidateFingerprint.of(report.candidate()), report);
        }

        List<BenchmarkCandidateReport> frontierReports = frontierFingerprints.stream()
                .map(reportsByFp::get)
                .filter(Objects::nonNull)
                .filter(BenchmarkCandidateReport::success)
                .filter(report -> report.measurement() != null)
                .sorted(Comparator.comparingDouble(report -> report.measurement().steadyStateStats().medianMs()))
                .limit(beamWidth)
                .toList();
        if (frontierReports.isEmpty()) {
            frontierFingerprints = List.of();
            return new SearchResult(List.of(), null);
        }

        LinkedHashMap<String, Candidate> next = new LinkedHashMap<>();
        List<String> nextFrontier = new ArrayList<>();
        for (BenchmarkCandidateReport report : frontierReports) {
            Candidate seed = report.candidate();
            String parentFp = CandidateFingerprint.of(seed);
            SearchTreeNode parent = nodesByFingerprint.get(parentFp);
            int accepted = 0;

            for (Candidate neighbor : refinable.neighbors(seed, context.request().workload())) {
                String fp = CandidateFingerprint.of(neighbor);
                if (seenFingerprints.contains(fp) || next.containsKey(fp)) {
                    continue;
                }
                next.put(fp, neighbor);
                nextFrontier.add(fp);
                nodesByFingerprint.putIfAbsent(fp, new SearchTreeNode(
                        fp,
                        neighbor.name(),
                        parentFp,
                        parent == null ? round : parent.depth() + 1,
                        round
                ));
                accepted++;
                if (accepted >= maxNeighborsPerNode || next.size() >= context.request().search().maxCandidates()) {
                    break;
                }
            }
            if (next.size() >= context.request().search().maxCandidates()) {
                break;
            }
        }

        frontierFingerprints = List.copyOf(nextFrontier);
        List<Candidate> selected = new ArrayList<>(next.values());
        Candidate preferred = selected.isEmpty() ? null : selected.getFirst();
        return new SearchResult(List.copyOf(selected), preferred);
    }

    public SearchTreeSnapshot snapshot() {
        return new SearchTreeSnapshot(List.copyOf(nodesByFingerprint.values()), frontierFingerprints);
    }
}
