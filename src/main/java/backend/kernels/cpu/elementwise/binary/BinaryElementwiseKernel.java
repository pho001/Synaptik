package backend.kernels.cpu.elementwise.binary;

import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
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

    default boolean supportsDirectF64() {
        return false;
    }

    default void runDirectF64(double[] left, double[] right, double[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT64 execution");
    }

    default boolean supportsDirectF32() {
        return false;
    }

    default void runDirectF32(float[] left, float[] right, float[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT32 execution");
    }

    default boolean supportsDirectBF16() {
        return false;
    }

    default void runDirectBF16(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedDispatchHints hints
    ) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct BF16 execution");
    }
}
