package operations.reduction;
import operations.Operation;

public final class reduceAll implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public reduceAll(int dimension) {
        this(dimension, false);
    }

    public reduceAll(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_ALL;
    }

    @Override
    public String getExpression() {
        return "reduceAll";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
