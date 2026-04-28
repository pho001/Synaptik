package operations.reduction;
import operations.Operation;

/**
 * Backpropagates through a maximum reduction along one dimension.
 *
 * <p>The descriptor records only the normalized axis; the paired tensor shapes
 * and any saved forward values are supplied by graph edges.</p>
 */
public final class reduceMaxGrad implements Operation {
    private final int dimension;

    /**
     * Creates a maximum-reduction gradient descriptor.
     *
     * @param dimension dimension along which the operation is applied
     */
    public reduceMaxGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_MAX_GRAD;
    }

    @Override
    public String getExpression() {
        return "reduce_max_grad";
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
