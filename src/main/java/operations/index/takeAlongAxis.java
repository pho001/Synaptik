package operations.index;
import operations.Operation;

/**
 * Takes values along one dimension using indices that are aligned with the input shape.
 *
 * <p>This descriptor matches take-along-axis semantics where indices select positions along the chosen axis.</p>
 */
public final class takeAlongAxis implements Operation {
    private final int dimension;

    /**
     * Creates an index operation descriptor for one dimension.
     *
     * @param dimension dimension along which indices are interpreted
     */
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

    /**
     * Returns the indexed dimension.
     *
     * @return dimension along which indices are interpreted
     */
    public int getDimension() {
        return dimension;
    }
}
