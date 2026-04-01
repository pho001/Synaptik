package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class mulScalar implements Operation{

    private final double scalar;
    private final float scalarF32;

    public mulScalar(double exponent) {
        this.scalar = exponent;
        this.scalarF32 = (float) exponent;
    }

    public mulScalar(float exponent) {
        this.scalarF32 = exponent;
        this.scalar = exponent;
    }

    @Override
    public OpType opType() {
        return OpType.MUL_SCALAR;
    }


    @Override
    public String getExpression() {
        return "*";
    }


    @Override
    public boolean isCheap() { return true;}

    public double getScalar() {
        return scalar;
    }

    public float getScalarF32() {
        return scalarF32;
    }
}
