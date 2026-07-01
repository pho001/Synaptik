package operations.index;

import operations.Operation;

/**
 * Functional tuple-index scatter using ONNX ScatterND shape semantics.
 */
public final class scatterNd implements Operation {
    private final ScatterReduction reduction;
    private final int batchDims;

    public scatterNd(ScatterReduction reduction) {
        this(reduction, 0);
    }

    public scatterNd(ScatterReduction reduction, int batchDims) {
        if (batchDims < 0) {
            throw new IllegalArgumentException("scatterNd batchDims must be non-negative.");
        }
        this.reduction = reduction == null ? ScatterReduction.NONE : reduction;
        this.batchDims = batchDims;
    }

    @Override
    public OpType opType() {
        return OpType.SCATTER_ND;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.SPECIAL;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.SPECIAL;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.MEDIUM;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "scatterNd(reduction=" + reduction + ",batchDims=" + batchDims + ")";
    }

    public ScatterReduction getReduction() {
        return reduction;
    }

    public int getBatchDims() {
        return batchDims;
    }
}
