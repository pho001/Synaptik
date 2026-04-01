package backend.kernels.cpu.fused;

import tensor.Tensor;

import java.util.List;

public interface CompiledFusedKernel {
    void applyRangeScalar(
            List<Tensor> inputs,
            Tensor out,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    );

    default void applyRangeVector(
            List<Tensor> inputs,
            Tensor out,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    ) {
        applyRangeScalar(inputs, out, startInclusive, endExclusive, options);
    }
}
