package operations;

public final class reduceMax implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public reduceMax(int dimension) {
        this(dimension, false);
    }

    public reduceMax(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_MAX;
    }

    @Override
    public String getExpression() {
        return "reduceMax";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
