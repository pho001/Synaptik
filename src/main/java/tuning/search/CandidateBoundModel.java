package tuning.search;

import tuning.benchmark.report.BenchmarkCandidateReport;

public interface CandidateBoundModel {
    double optimisticBound(
            BenchmarkCandidateReport report,
            SearchTreeNode node,
            CandidateScoreModel scoreModel,
            SearchContext context
    );
}
