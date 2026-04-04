package tuning.search;

import tuning.report.BenchmarkCandidateReport;

public final class MatMulBoundModel implements CandidateBoundModel {
    @Override
    public double optimisticBound(
            BenchmarkCandidateReport report,
            SearchTreeNode node,
            CandidateScoreModel scoreModel,
            SearchContext context
    ) {
        double score = scoreModel.score(report);
        if (context.request().workload().kind() != tuning.workload.WorkloadKind.MATMUL) {
            return score;
        }
        String name = report.candidate().name();
        if (name.contains("blasProvider=OPENBLAS_FFM")) {
            score *= 0.95d;
        }
        if (name.contains("blasThread=AUTO")) {
            score *= 0.99d;
        }
        if (name.contains("blasProvider=NONE")) {
            score *= 1.03d;
        }
        return score;
    }
}
