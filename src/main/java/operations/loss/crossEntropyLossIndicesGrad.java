package operations.loss;
import operations.Operation;

/**
 * Computes the gradient of index-target cross-entropy loss.
 *
 * <p>The gradient is shaped like the class-score input and uses the same class
 * dimension as the paired forward loss.</p>
 */
public final class crossEntropyLossIndicesGrad implements Operation {
    private final int classDimension;

    /**
     * Creates an index-target cross-entropy gradient descriptor.
     *
     * @param classDimension dimension containing class scores
     */
    public crossEntropyLossIndicesGrad(int classDimension) {
        this.classDimension = classDimension;
    }

    @Override
    public OpType opType() {
        return OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD;
    }

    @Override
    public String getExpression() {
        return "crossEntropyLossFromIndicesGrad";
    }

    /**
     * Returns the class-score dimension.
     *
     * @return class dimension used by the paired forward loss
     */
    public int getClassDimension() {
        return classDimension;
    }
}
