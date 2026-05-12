package operations.index;

import operations.Operation;

/**
 * Functional tuple-index gather using ONNX GatherND shape semantics.
 */
public final class gatherNd implements Operation {
    private final int batchDims;

    public gatherNd() {
        this(0);
    }

    public gatherNd(int batchDims) {
        if (batchDims < 0) {
            throw new IllegalArgumentException("gatherNd batchDims must be non-negative.");
        }
        this.batchDims = batchDims;
    }

    public int getBatchDims() {
        return batchDims;
    }

    @Override
    public OpType opType() {
        return OpType.GATHER_ND;
    }

    @Override
    public String getExpression() {
        return "gatherNd(batchDims=" + batchDims + ")";
    }
}
