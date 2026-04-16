package operations;

public final class softmaxGrad implements Operation {
    private final int dimension;

    public softmaxGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.SOFTMAX_GRAD;
    }

    @Override
    public String getExpression() {
        return "softmaxGrad";
    }

    public int getDimension() {
        return dimension;
    }
}
