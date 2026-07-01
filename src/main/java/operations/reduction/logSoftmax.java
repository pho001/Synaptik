package operations.reduction;
import operations.Operation;

/**
 * Normalizes values with log-softmax along one dimension.
 *
 * <p>The descriptor records only the normalized axis; the paired tensor shapes
 * and any saved forward values are supplied by graph edges.</p>
 */
public final class logSoftmax implements Operation {
    private final int dimension;

    /**
     * Creates a log-probabilities descriptor.
     *
     * @param dimension dimension along which the operation is applied
     */
    public logSoftmax(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public OpType opType() {
        return OpType.LOG_SOFTMAX;
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
        return OpComputationalCost.EXPENSIVE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "logSoftmax";
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
