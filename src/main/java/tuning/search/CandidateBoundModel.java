package tuning.search;

import tuning.report.BenchmarkCandidateReport;

public interface CandidateBoundModel {
    double optimisticBound(BenchmarkCandidateReport report, SearchTreeNode node, CandidateScoreModel scoreModel);
}
