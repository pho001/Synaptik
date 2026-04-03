package operations;

public final class reduceAny implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public reduceAny(int dimension) {
        this(dimension, false);
    }

    public reduceAny(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_ANY;
    }

    @Override
    public String getExpression() {
        return "reduceAny";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
