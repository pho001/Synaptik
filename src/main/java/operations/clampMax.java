package operations;

public final class clampMax implements Operation {
    private final double maxValue;
    private final float maxValueF32;

    public clampMax(double maxValue) {
        this.maxValue = maxValue;
        this.maxValueF32 = (float) maxValue;
    }

    public clampMax(float maxValue) {
        this.maxValue = maxValue;
        this.maxValueF32 = maxValue;
    }

    @Override
    public OpType opType() {
        return OpType.CLAMP_MAX;
    }

    @Override
    public String getExpression() {
        return "clampMax(" + maxValue + ")";
    }

    public double getMaxValue() {
        return maxValue;
    }

    public float getMaxValueF32() {
        return maxValueF32;
    }
}
