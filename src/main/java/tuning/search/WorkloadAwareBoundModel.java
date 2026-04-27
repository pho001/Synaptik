package tuning.search;

import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.workload.WorkloadKind;

public final class WorkloadAwareBoundModel implements CandidateBoundModel {
    private final CandidateBoundModel generic;
    private final CandidateBoundModel conv2d;
    private final CandidateBoundModel matmul;
    private final CandidateBoundModel transformer;

    public WorkloadAwareBoundModel() {
        this(new ParentScoreBoundModel(), new Conv2dBoundModel(), new MatMulBoundModel(), new TransformerHotPathBoundModel());
    }

    public WorkloadAwareBoundModel(
            CandidateBoundModel generic,
            CandidateBoundModel conv2d,
            CandidateBoundModel matmul,
            CandidateBoundModel transformer
    ) {
        this.generic = generic;
        this.conv2d = conv2d;
        this.matmul = matmul;
        this.transformer = transformer;
    }

    @Override
    public double optimisticBound(
            BenchmarkCandidateReport report,
            SearchTreeNode node,
            CandidateScoreModel scoreModel,
            SearchContext context
    ) {
        WorkloadKind kind = context.request().workload().kind();
        return switch (kind) {
            case CONV2D -> conv2d.optimisticBound(report, node, scoreModel, context);
            case MATMUL -> matmul.optimisticBound(report, node, scoreModel, context);
            case TRANSFORMER_HOT_PATH -> transformer.optimisticBound(report, node, scoreModel, context);
            default -> generic.optimisticBound(report, node, scoreModel, context);
        };
    }
}
