package operations.reduction;
import operations.Operation;

/**
 * Backpropagates through a minimum reduction along one dimension.
 *
 * <p>The descriptor records only the normalized axis; the paired tensor shapes
 * and any saved forward values are supplied by graph edges.</p>
 */
public final class reduceMinGrad implements Operation {
    private final int dimension;

    /**
     * Creates a minimum-reduction gradient descriptor.
     *
     * @param dimension dimension along which the operation is applied
     */
    public reduceMinGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.REDUCE_MIN_GRAD;
    }

    @Override
    public String getExpression() {
        return "reduce_min_grad";
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
