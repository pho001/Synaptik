package backend.cpu.kernels.elementwise.unary;

import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

import java.lang.foreign.MemorySegment;

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

    default boolean supportsDirectF64() {
        return false;
    }

    default void runDirectF64(double[] in, double[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT64 execution");
    }

    default boolean supportsDirectF32() {
        return false;
    }

    default void runDirectF32(float[] in, float[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT32 execution");
    }

    default boolean supportsDirectBF16() {
        return false;
    }

    default void runDirectBF16(short[] in, float[] continuation, short[] out, ResolvedDispatchHints hints) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct BF16 execution");
    }

    default void runSegmentF64(MemorySegment in, MemorySegment out, int start, int end) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT64 segment execution");
    }

    default void runSegmentF32(MemorySegment in, MemorySegment out, int start, int end) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct FLOAT32 segment execution");
    }

    default void runSegmentBF16(MemorySegment in, MemorySegment out, int start, int end) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support direct BF16 segment execution");
    }
}
