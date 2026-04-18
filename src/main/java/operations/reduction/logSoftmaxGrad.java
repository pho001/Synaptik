package operations.reduction;
import operations.Operation;

public final class logSoftmaxGrad implements Operation {
    private final int dimension;

    public logSoftmaxGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.LOG_SOFTMAX_GRAD;
    }

    @Override
    public String getExpression() {
        return "logSoftmaxGrad";
    }

    public int getDimension() {
        return dimension;
    }
}
