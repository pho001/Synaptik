package operations.reduction;

import operations.Operation;

/**
 * Sums tensor values along one dimension.
 *
 * <p>The reduced axis is removed by default. When {@code keepDims} is true,
 * the axis is retained with extent one so downstream broadcasting can align
 * with the original rank.</p>
 */
public final class sum implements Operation {
    private final int dimension;
    private final boolean keepDims;

    /**
     * Creates a descriptor that reduces the given dimension and removes it.
     *
     * @param dimension dimension to reduce
     */
    public sum(int dimension) {
        this(dimension, false);
    }

    /**
     * Creates a descriptor that reduces the given dimension.
     *
     * @param dimension dimension to reduce
     * @param keepDims whether to retain the reduced axis with extent one
     */
    public sum(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.SUM;
    }

    @Override
    public String getExpression() {
        return "sum";
    }

    /**
     * Returns the reduced dimension.
     *
     * @return dimension supplied by the tensor front end
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Indicates whether the reduced axis is retained with extent one.
     *
     * @return {@code true} when the output keeps the reduced dimension
     */
    public boolean keepDims() {
        return keepDims;
    }
}
