package operations.reduction;
import operations.Operation;

public final class logSoftmax implements Operation {
    private final int dimension;

    public logSoftmax(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.LOG_SOFTMAX;
    }

    @Override
    public String getExpression() {
        return "logSoftmax";
    }

    public int getDimension() {
        return dimension;
    }
}
