package operations.reduction;
import operations.Operation;

public final class reduceMinGrad implements Operation {
    private final int dimension;

    public reduceMinGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_MIN_GRAD;
    }

    @Override
    public String getExpression() {
        return "reduce_min_grad";
    }

    public int getDimension() {
        return dimension;
    }
}
