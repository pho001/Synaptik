package operations.index;
import operations.Operation;

/**
 * Accumulates the gradient of a gather operation back into the source tensor shape.
 *
 * <p>Repeated indices contribute additively to the same source position.</p>
 */
public final class gatherGrad implements Operation {
    private final int dimension;

    /**
     * Creates an index operation descriptor for one dimension.
     *
     * @param dimension dimension along which indices are interpreted
     */
    public gatherGrad(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.GATHER_GRAD;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.SPECIAL;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.SPECIAL;
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
        return "gatherGrad";
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
