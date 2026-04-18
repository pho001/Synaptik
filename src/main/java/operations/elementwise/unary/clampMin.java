package operations.elementwise.unary;
import operations.Operation;

public final class clampMin implements Operation {
    private final double minValue;
    private final float minValueF32;

    public clampMin(double minValue) {
        this.minValue = minValue;
        this.minValueF32 = (float) minValue;
    }

    public clampMin(float minValue) {
        this.minValue = minValue;
        this.minValueF32 = minValue;
    }

    @Override
    public OpType opType() {
        return OpType.CLAMP_MIN;
    }

    @Override
    public String getExpression() {
        return "clampMin(" + minValue + ")";
    }

    public double getMinValue() {
        return minValue;
    }

    public float getMinValueF32() {
        return minValueF32;
    }
}
