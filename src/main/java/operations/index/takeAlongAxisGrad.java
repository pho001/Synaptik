package operations.index;
import operations.Operation;

public final class takeAlongAxisGrad implements Operation {
    private final int dimension;

    public takeAlongAxisGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.TAKE_ALONG_AXIS_GRAD;
    }

    @Override
    public String getExpression() {
        return "takeAlongAxisGrad";
    }

    public int getDimension() {
        return dimension;
    }
}
