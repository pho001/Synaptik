package operations.index;
import operations.Operation;

public final class gatherGrad implements Operation {
    private final int dimension;

    public gatherGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.GATHER_GRAD;
    }

    @Override
    public String getExpression() {
        return "gatherGrad";
    }

    public int getDimension() {
        return dimension;
    }
}
