package operations.elementwise.unary;
import operations.Operation;

/**
 * Clamps each element to be no greater than a scalar maximum.
 *
 * <p>The output shape matches the input shape; numeric dtype behavior is
 * resolved by the tensor/backend execution contract.</p>
 */
public final class clampMax implements Operation {
    private final double maxValue;
    private final float maxValueF32;

    /**
     * Creates the descriptor with a double-precision maximum bound.
     *
     * @param maxValue maximum bound to apply
     */
    public clampMax(double maxValue) {
        this.maxValue = maxValue;
        this.maxValueF32 = (float) maxValue;
    }

    /**
     * Creates the descriptor with a single-precision maximum bound.
     *
     * @param maxValue maximum bound to apply
     */
    public clampMax(float maxValue) {
        this.maxValue = maxValue;
        this.maxValueF32 = maxValue;
    }

    @Override
    public OpType opType() {
        return OpType.CLAMP_MAX;
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
    public OpControlTrait controlTrait() {
        return OpControlTrait.BRANCHLESS;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "clampMax(" + maxValue + ")";
    }

    /**
     * Returns the maximum bound as a double.
     *
     * @return double-precision maximum bound
     */
    public double getMaxValue() {
        return maxValue;
    }

    /**
     * Returns the maximum bound as a float.
     *
     * @return single-precision maximum bound
     */
    public float getMaxValueF32() {
        return maxValueF32;
    }
}
