package operations.linalg;
import operations.Operation;

/**
 * Affine linear projection descriptor.
 *
 * <p>The operation applies a weight matrix to the final input dimension and may
 * add a bias vector. Shape validation and dtype promotion are handled by the
 * tensor front end and backend kernels.</p>
 */
public final class linear implements Operation {
    private final boolean hasBias;

    /**
     * Creates a linear descriptor.
     *
     * @param hasBias whether the operation includes a bias input
     */
    public linear(boolean hasBias) {
        this.hasBias = hasBias;
    }

    /**
     * Indicates whether this linear operation includes bias addition.
     *
     * @return {@code true} when a bias input is expected
     */
    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public OpType opType() {
        return OpType.LINEAR;
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
        return OpSemanticFamily.LINEAR_ALGEBRA;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.EXPENSIVE;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.NONE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return hasBias ? "linear+bias" : "linear";
    }
}
