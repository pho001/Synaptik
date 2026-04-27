package tuning.search;

import tuning.benchmark.report.BenchmarkCandidateReport;

public final class Conv2dBoundModel implements CandidateBoundModel {
    @Override
    public double optimisticBound(
            BenchmarkCandidateReport report,
            SearchTreeNode node,
            CandidateScoreModel scoreModel,
            SearchContext context
    ) {
        double score = scoreModel.score(report);
        if (context.request().workload().kind() != tuning.workload.WorkloadKind.CONV2D) {
            return score;
        }
        String name = report.candidate().name();
        if (name.contains("conv2dLowering=HEURISTIC")) {
            return score * 0.96d;
        }
        if (name.contains("conv2dLowering=ALWAYS")) {
            return score * 1.03d;
        }
        if (name.contains("conv2dLowering=OFF")) {
            return score * 1.06d;
        }
        return score;
    }
}
