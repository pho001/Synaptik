package operations.index;
import operations.Operation;

/**
 * Accumulates the gradient of take-along-axis back into the source tensor shape.
 *
 * <p>Repeated selected positions contribute additively during gradient scatter.</p>
 */
public final class takeAlongAxisGrad implements Operation {
    private final int dimension;

    /**
     * Creates an index operation descriptor for one dimension.
     *
     * @param dimension dimension along which indices are interpreted
     */
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

    /**
     * Returns the indexed dimension.
     *
     * @return dimension along which indices are interpreted
     */
    public int getDimension() {
        return dimension;
    }
}
