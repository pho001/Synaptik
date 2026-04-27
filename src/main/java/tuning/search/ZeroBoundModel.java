package tuning.search;

import tuning.benchmark.report.BenchmarkCandidateReport;

public final class ZeroBoundModel implements CandidateBoundModel {
    @Override
    public double optimisticBound(BenchmarkCandidateReport report, SearchTreeNode node, CandidateScoreModel scoreModel, SearchContext context) {
        return 0.0d;
    }
}
