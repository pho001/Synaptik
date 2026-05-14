package operations.reduction;

import operations.Operation;

/**
 * Returns the first maximum-value index along one dimension.
 */
public final class argMax implements Operation {
    private final int dimension;
    private final boolean keepDims;
    private final ArgMaxTiePolicy tiePolicy;

    public argMax(int dimension, boolean keepDims) {
        this(dimension, keepDims, ArgMaxTiePolicy.FIRST_INDEX);
    }

    public argMax(int dimension, boolean keepDims, ArgMaxTiePolicy tiePolicy) {
        this.dimension = dimension;
        this.keepDims = keepDims;
        this.tiePolicy = tiePolicy == null ? ArgMaxTiePolicy.FIRST_INDEX : tiePolicy;
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

    public ArgMaxTiePolicy tiePolicy() {
        return tiePolicy;
    }
}
