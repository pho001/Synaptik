package operations;

public final class reduceMaxGrad implements Operation {
    private final int dimension;

    public reduceMaxGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_MAX_GRAD;
    }

    @Override
    public String getExpression() {
        return "reduce_max_grad";
    }

    public int getDimension() {
        return dimension;
    }
}
