package tuning.search;

import tuning.report.BenchmarkCandidateReport;

public final class ParentScoreBoundModel implements CandidateBoundModel {
    @Override
    public double optimisticBound(BenchmarkCandidateReport report, SearchTreeNode node, CandidateScoreModel scoreModel, SearchContext context) {
        return scoreModel.score(report);
    }
}
