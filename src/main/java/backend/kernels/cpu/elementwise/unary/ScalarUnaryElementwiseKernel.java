package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
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

    default boolean supportsDirectF64() {
        return false;
    }

    default void runDirectF64(double[] in, double parameter, double[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT64 execution");
    }

    default boolean supportsDirectF32() {
        return false;
    }

    default void runDirectF32(float[] in, float parameter, float[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT32 execution");
    }

    default boolean supportsDirectBF16() {
        return false;
    }

    default void runDirectBF16(short[] in, float[] continuation, float parameter, short[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct BF16 execution");
    }
}
