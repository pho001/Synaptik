package operations.reduction;
import operations.Operation;

/**
 * Computes the gradient of log-softmax along one dimension.
 *
 * <p>The descriptor records only the normalized axis; the paired tensor shapes
 * and any saved forward values are supplied by graph edges.</p>
 */
public final class logSoftmaxGrad implements Operation {
    private final int dimension;

    /**
     * Creates a log-softmax gradient descriptor.
     *
     * @param dimension dimension along which the operation is applied
     */
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

    /**
     * Returns the operation dimension.
     *
     * @return dimension along which this descriptor operates
     */
    public int getDimension() {
        return dimension;
    }
}
