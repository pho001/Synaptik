package operations.elementwise.unary;

import operations.Operation;

public final class pow implements Operation {
    private final double exponent;
    private final float exponentF32;

    public pow(double exponent) {
        this.exponent = exponent;
        this.exponentF32 = (float) exponent;
    }

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

    public double getExponent() {
        return exponent;
    }

    public float getExponentF32() {
        return exponentF32;
    }
}
