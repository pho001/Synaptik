package operations.index;
import operations.Operation;

/**
 * Gathers values from a source tensor along one dimension using index tensor entries.
 *
 * <p>For example, gathering along the class dimension can select one value per row from a batched score tensor.</p>
 */
public final class gather implements Operation {
    private final int dimension;

    /**
     * Creates an index operation descriptor for one dimension.
     *
     * @param dimension dimension along which indices are interpreted
     */
    public gather(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.GATHER;
    }

    @Override
    public String getExpression() {
        return "gather";
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
