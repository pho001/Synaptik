package operations.layout;
import operations.Operation;

/**
 * Selects one index from a tensor dimension and removes that dimension.
 *
 * <p>For example, selecting index {@code i} along dimension {@code d} produces
 * the slice at that position with rank reduced by one.</p>
 */
public final class select implements Operation {
    private final int dimension;
    private final int index;

    /**
     * Creates a select descriptor.
     *
     * @param dimension dimension to index
     * @param index position selected from that dimension
     */
    public select(int dimension, int index) {
        this.dimension = dimension;
        this.index = index;
    }

    /**
     * Returns the selected dimension.
     *
     * @return dimension to index
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Returns the selected position.
     *
     * @return index selected from the dimension
     */
    public int getIndex() {
        return index;
    }

    @Override
    public OpType opType() {
        return OpType.SELECT;
    }

    @Override
    public String getExpression() {
        return "select(" + dimension + "," + index + ")";
    }
}
