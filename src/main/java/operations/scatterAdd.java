package operations;

public final class scatterAdd implements Operation {
    private final int dimension;

    public scatterAdd(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.SCATTER_ADD;
    }

    @Override
    public String getExpression() {
        return "scatterAdd";
    }

    public int getDimension() {
        return dimension;
    }
}
