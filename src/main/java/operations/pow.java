package operations;

import java.util.List;
import tensor.Tensor;
import backend.ComputeBackend;

public class pow implements Operation {

    private final double exponent;
    private final float exponentF32;

    //default implementation - CPU
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
