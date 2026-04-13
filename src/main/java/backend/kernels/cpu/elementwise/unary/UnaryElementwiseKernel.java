package backend.kernels.cpu.elementwise.unary;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

public interface UnaryElementwiseKernel {
    double applyF64(double value);

    float applyF32(float value);

    float applyBF16(float value);

    default boolean supportsVectorF64() {
        return false;
    }

    default DoubleVector applyVectorF64(DoubleVector value) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT64 vectors");
    }

    default boolean supportsVectorF32() {
        return false;
    }

    default FloatVector applyVectorF32(FloatVector value) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT32 vectors");
    }
}
