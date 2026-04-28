package operations.loss;
import operations.Operation;

/**
 * Computes cross-entropy loss from class-probability or logit tensors.
 *
 * <p>The class dimension identifies the axis containing class scores; target
 * shape and reduction behavior are supplied by the tensor-level API.</p>
 */
public final class crossEntropyLoss implements Operation {
    private final int classDimension;

    /**
     * Creates a cross-entropy descriptor.
     *
     * @param classDimension dimension containing class scores
     */
    public crossEntropyLoss(int classDimension) {
        this.classDimension = classDimension;
    }

    @Override
    public OpType opType() {
        return OpType.CROSS_ENTROPY_LOSS;
    }

    @Override
    public String getExpression() {
        return "crossEntropyLoss";
    }

    /**
     * Returns the class-score dimension.
     *
     * @return class dimension supplied by the tensor front end
     */
    public int getClassDimension() {
        return classDimension;
    }
}
