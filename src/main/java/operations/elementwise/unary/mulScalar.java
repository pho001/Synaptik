package operations.elementwise.unary;

import operations.Operation;

public final class mulScalar implements Operation {
    private final double scalar;
    private final float scalarF32;

    public mulScalar(double scalar) {
        this.scalar = scalar;
        this.scalarF32 = (float) scalar;
    }

    public mulScalar(float scalar) {
        this.scalarF32 = scalar;
        this.scalar = scalar;
    }

    @Override
    public OpType opType() {
        return OpType.MUL_SCALAR;
    }

    @Override
    public String getExpression() {
        return "mulScalar(" + scalar + ")";
    }

    @Override
    public boolean isCheap() {
        return true;
    }

    public double getScalar() {
        return scalar;
    }

    public float getScalarF32() {
        return scalarF32;
    }
}
