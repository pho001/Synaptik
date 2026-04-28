package operations.elementwise.unary;

import operations.Operation;

/**
 * Raises each element to a scalar exponent.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class pow implements Operation {
    private final double exponent;
    private final float exponentF32;

    /**
     * Creates the descriptor with a double-precision exponent.
     *
     * @param exponent exponent to apply
     */
    public pow(double exponent) {
        this.exponent = exponent;
        this.exponentF32 = (float) exponent;
    }

    /**
     * Creates the descriptor with a single-precision exponent.
     *
     * @param exponent exponent to apply
     */
    public pow(float exponent) {
        this.exponentF32 = exponent;
        this.exponent = exponent;
    }

    @Override
    public OpType opType() {
        return OpType.POW;
    }

    @Override
    public String getExpression() {
        return "pow(" + exponent + ")";
    }

    /**
     * Returns the exponent as a double.
     *
     * @return double-precision exponent
     */
    public double getExponent() {
        return exponent;
    }

    /**
     * Returns the exponent as a float.
     *
     * @return single-precision exponent
     */
    public float getExponentF32() {
        return exponentF32;
    }
}
