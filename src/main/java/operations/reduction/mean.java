package operations.reduction;
import operations.Operation;

public final class mean implements Operation {
    private final int dimension;
    private final boolean keepDims;

    public mean(int dimension) {
        this(dimension, false);
    }

    public mean(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.MEAN;
    }

    @Override
    public String getExpression() {
        return "mean";
    }

    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }
}
