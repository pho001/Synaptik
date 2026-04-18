package operations.index;
import operations.Operation;

public final class takeAlongAxis implements Operation {
    private final int dimension;

    public takeAlongAxis(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.TAKE_ALONG_AXIS;
    }

    @Override
    public String getExpression() {
        return "takeAlongAxis";
    }

    public int getDimension() {
        return dimension;
    }
}
