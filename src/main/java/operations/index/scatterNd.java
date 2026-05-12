package operations.index;

import operations.Operation;

/**
 * Functional tuple-index scatter using ONNX ScatterND shape semantics.
 */
public final class scatterNd implements Operation {
    private final ScatterReduction reduction;

    public scatterNd(ScatterReduction reduction) {
        this.reduction = reduction == null ? ScatterReduction.NONE : reduction;
    }

    @Override
    public OpType opType() {
        return OpType.SCATTER_ND;
    }

    @Override
    public String getExpression() {
        return "scatterNd(reduction=" + reduction + ")";
    }

    public ScatterReduction getReduction() {
        return reduction;
    }
}
