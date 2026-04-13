package backend.kernels.cpu.elementwise.unary;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

public interface ScalarUnaryElementwiseKernel {
    double applyF64(double value, double parameter);

    float applyF32(float value, float parameter);

    float applyBF16(float value, float parameter);

    default boolean supportsVectorF64() {
        return false;
    }

    default DoubleVector applyVectorF64(DoubleVector value, DoubleVector parameter) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT64 vectors");
    }

    default boolean supportsVectorF32() {
        return false;
    }

    default FloatVector applyVectorF32(FloatVector value, FloatVector parameter) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT32 vectors");
    }
}
