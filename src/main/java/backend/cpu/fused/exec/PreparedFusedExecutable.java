package backend.cpu.fused.exec;

import backend.cpu.kernels.CpuKernelContext;
import tensor.Tensor;

import java.util.List;

/**
 * Prepared executable for a CPU fused operation.
 *
 * <p>Implementations evaluate a half-open output range into {@code out}. Vector
 * execution is optional; the default vector path delegates to scalar execution.</p>
 */
public interface PreparedFusedExecutable {
    /**
     * Applies the fused operation over {@code [startInclusive, endExclusive)} using scalar code.
     */
    void applyRangeScalar(
            List<Tensor> inputs,
            Tensor out,
            CpuKernelContext context,
            int startInclusive,
            int endExclusive
    );

    /**
     * Applies the fused operation over {@code [startInclusive, endExclusive)} using vector code when available.
     */
    default void applyRangeVector(
            List<Tensor> inputs,
            Tensor out,
            CpuKernelContext context,
            int startInclusive,
            int endExclusive
    ) {
        applyRangeScalar(inputs, out, context, startInclusive, endExclusive);
    }
}
