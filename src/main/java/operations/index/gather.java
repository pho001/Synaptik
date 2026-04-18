package operations.index;
import operations.Operation;

public final class gather implements Operation {
    private final int dimension;

    public gather(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.GATHER;
    }

    @Override
    public String getExpression() {
        return "gather";
    }

    public int getDimension() {
        return dimension;
    }
}
