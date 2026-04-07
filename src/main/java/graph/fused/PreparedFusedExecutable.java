package graph.fused;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.fused.FusedExecutionOptions;
import tensor.Tensor;

import java.util.List;

public interface PreparedFusedExecutable {
    void applyRangeScalar(
            List<Tensor> inputs,
            Tensor out,
            CpuKernelContext context,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    );

    default void applyRangeVector(
            List<Tensor> inputs,
            Tensor out,
            CpuKernelContext context,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    ) {
        applyRangeScalar(inputs, out, context, startInclusive, endExclusive, options);
    }
}
