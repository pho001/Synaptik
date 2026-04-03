package operations;

public final class select implements Operation {
    private final int dimension;
    private final int index;

    public select(int dimension, int index) {
        this.dimension = dimension;
        this.index = index;
    }

    public int getDimension() {
        return dimension;
    }

    public int getIndex() {
        return index;
    }

    @Override
    public OpType opType() {
        return OpType.SELECT;
    }

    @Override
    public String getExpression() {
        return "select(" + dimension + "," + index + ")";
    }
}
