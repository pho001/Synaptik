package operations.index;
import operations.Operation;

/**
 * Adds update values into a destination tensor at indexed positions along one dimension.
 *
 * <p>For example, scatter-add can rebuild source gradients from gathered updates by summing repeated indices.</p>
 */
public final class scatterAdd implements Operation {
    private final int dimension;

    /**
     * Creates an index operation descriptor for one dimension.
     *
     * @param dimension dimension along which indices are interpreted
     */
    public scatterAdd(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.SCATTER_ADD;
    }

    @Override
    public String getExpression() {
        return "scatterAdd";
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
