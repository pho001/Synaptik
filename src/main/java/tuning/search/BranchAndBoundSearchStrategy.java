package tuning.search;

import tuning.candidate.Candidate;
import tuning.candidate.ExecutableProfileFingerprint;
import tuning.candidate.RefinableCandidateSpace;
import tuning.benchmark.report.BenchmarkCandidateReport;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class BranchAndBoundSearchStrategy extends AbstractTreeSearchStrategy {
    private final CandidateScoreModel scoreModel;
    private final CandidateBoundModel boundModel;
    private final int beamWidth;
    private final int maxNeighborsPerNode;
    private final List<String> prunedFingerprints = new ArrayList<>();

    public BranchAndBoundSearchStrategy(
            SearchStrategy seedStrategy,
            CandidateScoreModel scoreModel,
            CandidateBoundModel boundModel,
            int beamWidth,
            int maxNeighborsPerNode
    ) {
        super(seedStrategy);
        this.scoreModel = Objects.requireNonNull(scoreModel, "scoreModel cannot be null");
        this.boundModel = Objects.requireNonNull(boundModel, "boundModel cannot be null");
        this.beamWidth = Math.max(1, beamWidth);
        this.maxNeighborsPerNode = Math.max(1, maxNeighborsPerNode);
    }

    @Override
    public SearchResult search(SearchContext context) {
        prunedFingerprints.clear();
        return super.search(context);
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
            reportsByFp.put(ExecutableProfileFingerprint.of(report.candidate()), report);
        }

        double bestScore = evaluatedSoFar.stream()
                .filter(BenchmarkCandidateReport::success)
                .filter(report -> report.measurement() != null)
                .mapToDouble(scoreModel::score)
                .min()
                .orElse(Double.POSITIVE_INFINITY);

        List<BenchmarkCandidateReport> expandable = new ArrayList<>();
        for (String fp : frontierFingerprints) {
            BenchmarkCandidateReport report = reportsByFp.get(fp);
            if (report == null || !report.success() || report.measurement() == null) {
                continue;
            }
            SearchTreeNode node = nodesByFingerprint.get(fp);
            double bound = boundModel.optimisticBound(report, node, scoreModel, context);
            if (bound <= bestScore) {
                expandable.add(report);
            } else {
                prunedFingerprints.add(fp);
            }
        }
        expandable.sort(Comparator.comparingDouble(scoreModel::score));
        expandable = expandable.stream().limit(beamWidth).toList();
        if (expandable.isEmpty()) {
            frontierFingerprints = List.of();
            return new SearchResult(List.of(), null);
        }

        LinkedHashMap<String, Candidate> next = new LinkedHashMap<>();
        List<String> nextFrontier = new ArrayList<>();
        for (BenchmarkCandidateReport report : expandable) {
            Candidate seed = report.candidate();
            String parentFp = ExecutableProfileFingerprint.of(seed);
            int accepted = 0;
            for (Candidate neighbor : refinable.neighbors(seed, context.request().workload())) {
                String fp = ExecutableProfileFingerprint.of(neighbor);
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
            if (next.size() >= context.request().search().maxCandidates()) {
                break;
            }
        }

        frontierFingerprints = List.copyOf(nextFrontier);
        List<Candidate> selected = new ArrayList<>(next.values());
        return new SearchResult(List.copyOf(selected), selected.isEmpty() ? null : selected.getFirst());
    }

    public SearchTreeSnapshot snapshot() {
        return super.snapshot();
    }

    public List<String> prunedFingerprints() {
        return List.copyOf(prunedFingerprints);
    }
}
