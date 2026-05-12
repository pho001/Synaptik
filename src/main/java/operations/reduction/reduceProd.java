package operations.reduction;

import operations.Operation;

/**
 * Multiplies tensor values along one dimension.
 */
public final class reduceProd implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public reduceProd(int dimension) {
        this(dimension, false);
    }

    public reduceProd(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_PROD;
    }

    @Override
    public String getExpression() {
        return "reduceProd";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
