package operations.reduction;

import operations.Operation;

/**
 * Returns the first maximum-value index along one dimension.
 */
public final class argMax implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public argMax(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.ARGMAX;
    }

    @Override
    public String getExpression() {
        return "argMax";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
