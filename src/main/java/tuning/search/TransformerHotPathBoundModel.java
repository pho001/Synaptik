package tuning.search;

import tuning.report.BenchmarkCandidateReport;

public final class TransformerHotPathBoundModel implements CandidateBoundModel {
    @Override
    public double optimisticBound(
            BenchmarkCandidateReport report,
            SearchTreeNode node,
            CandidateScoreModel scoreModel,
            SearchContext context
    ) {
        double score = scoreModel.score(report);
        if (context.request().workload().kind() != tuning.workload.WorkloadKind.TRANSFORMER_HOT_PATH) {
            return score;
        }
        String name = report.candidate().name();
        if (name.contains("attentionMatMul=FORCE_OFF")) {
            score *= 1.07d;
        } else if (name.contains("attentionMatMul=FORCE_ON")) {
            score *= 0.98d;
        } else if (name.contains("attentionMatMul=AUTO")) {
            score *= 0.99d;
        }
        if (name.contains("blasProvider=OPENBLAS_FFM")) {
            score *= 0.97d;
        }
        if (name.contains("vectorPolicies=FORCE_ON")) {
            score *= 0.99d;
        }
        if (node != null && node.depth() > 1) {
            score *= 1.01d;
        }
        return score;
    }
}
