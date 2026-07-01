package operations.elementwise.unary;
import operations.Operation;

/**
 * Clamps each element to be no less than a scalar minimum.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class clampMin implements Operation {
    private final double minValue;
    private final float minValueF32;

    /**
     * Creates the descriptor with a double-precision minimum bound.
     *
     * @param minValue minimum bound to apply
     */
    public clampMin(double minValue) {
        this.minValue = minValue;
        this.minValueF32 = (float) minValue;
    }

    /**
     * Creates the descriptor with a single-precision minimum bound.
     *
     * @param minValue minimum bound to apply
     */
    public clampMin(float minValue) {
        this.minValue = minValue;
        this.minValueF32 = minValue;
    }

    @Override
    public OpType opType() {
        return OpType.CLAMP_MIN;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.ELEMENT_WISE;
    }

    @Override
    public boolean isFusable() {
        return true;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.ARITHMETIC;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.CHEAP;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "clampMin(" + minValue + ")";
    }

    /**
     * Returns the minimum bound as a double.
     *
     * @return double-precision minimum bound
     */
    public double getMinValue() {
        return minValue;
    }

    /**
     * Returns the minimum bound as a float.
     *
     * @return single-precision minimum bound
     */
    public float getMinValueF32() {
        return minValueF32;
    }
}
