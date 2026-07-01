package operations.elementwise.unary;

import operations.Operation;

/**
 * Multiplies every element by a scalar constant.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class mulScalar implements Operation {
    private final double scalar;
    private final float scalarF32;

    /**
     * Creates the descriptor with a double-precision scalar.
     *
     * @param scalar scalar to apply
     */
    public mulScalar(double scalar) {
        this.scalar = scalar;
        this.scalarF32 = (float) scalar;
    }

    /**
     * Creates the descriptor with a single-precision scalar.
     *
     * @param scalar scalar to apply
     */
    public mulScalar(float scalar) {
        this.scalarF32 = scalar;
        this.scalar = scalar;
    }

    @Override
    public OpType opType() {
        return OpType.MUL_SCALAR;
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
        return "mulScalar(" + scalar + ")";
    }

    /**
     * Returns the scalar as a double.
     *
     * @return double-precision scalar
     */
    public double getScalar() {
        return scalar;
    }

    /**
     * Returns the scalar as a float.
     *
     * @return single-precision scalar
     */
    public float getScalarF32() {
        return scalarF32;
    }
}
