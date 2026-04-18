package operations.reduction;

import operations.Operation;

public final class sum implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public sum(int dimension) {
        this(dimension, false);
    }

    public sum(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.SUM;
    }

    @Override
    public String getExpression() {
        return "sum";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
