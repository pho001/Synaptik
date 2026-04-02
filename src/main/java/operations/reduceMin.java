package operations;

public final class reduceMin implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public reduceMin(int dimension) {
        this(dimension, false);
    }

    public reduceMin(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_MIN;
    }

    @Override
    public String getExpression() {
        return "reduceMin";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
