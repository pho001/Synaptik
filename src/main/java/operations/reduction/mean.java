package operations.reduction;
import operations.Operation;

/**
 * Averages tensor values along one dimension.
 *
 * <p>The reduced axis is removed by default. When {@code keepDims} is true,
 * the axis is retained with extent one so downstream broadcasting can align
 * with the original rank.</p>
 */
public final class mean implements Operation {
    private final int dimension;
    private final boolean keepDims;

    /**
     * Creates a descriptor that reduces the given dimension and removes it.
     *
     * @param dimension dimension to reduce
     */
    public mean(int dimension) {
        this(dimension, false);
    }

    /**
     * Creates a descriptor that reduces the given dimension.
     *
     * @param dimension dimension to reduce
     * @param keepDims whether to retain the reduced axis with extent one
     */
    public mean(int dimension, boolean keepDims) {
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    @Override
    public OpType opType() {
        return OpType.MEAN;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.REDUCTION;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.REDUCTION;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.MEDIUM;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "mean";
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
