package operations;

public final class softmax implements Operation {
    private final int dimension;

    public softmax(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.SOFTMAX;
    }

    @Override
    public String getExpression() {
        return "softmax";
    }

    public int getDimension() {
        return dimension;
    }
}
