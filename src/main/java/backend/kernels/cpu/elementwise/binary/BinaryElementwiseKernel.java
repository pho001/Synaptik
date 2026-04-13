package backend.kernels.cpu.elementwise.binary;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

public interface BinaryElementwiseKernel {
    double applyF64(double left, double right);

    float applyF32(float left, float right);

    float applyBF16(float left, float right);

    default boolean supportsVectorF64() {
        return false;
    }

    default DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT64 vectors");
    }

    default boolean supportsVectorF32() {
        return false;
    }

    default FloatVector applyVectorF32(FloatVector left, FloatVector right) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT32 vectors");
    }
}
