package operations.loss;
import operations.Operation;

/**
 * Computes negative log-likelihood loss along a class dimension.
 *
 * <p>The descriptor assumes class scores are already in log-probability form;
 * target shape and reduction semantics are supplied by tensor graph edges.</p>
 */
public final class nllLoss implements Operation {
    private final int classDimension;

    /**
     * Creates an NLL loss descriptor.
     *
     * @param classDimension dimension containing class log-probabilities
     */
    public nllLoss(int classDimension) {
        this.classDimension = classDimension;
    }

    @Override
    public OpType opType() {
        return OpType.NLL_LOSS;
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
        return "nllLoss";
    }

    /**
     * Returns the class log-probability dimension.
     *
     * @return class dimension supplied by the tensor front end
     */
    public int getClassDimension() {
        return classDimension;
    }
}
