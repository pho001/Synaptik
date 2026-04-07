package graph.fused.vector;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.Float32FusedExecutor;
import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.fused.PreparedFusedExecutable;
import operations.FusedOperation;
import tensor.Tensor;

import java.util.List;

public final class Float32PreparedFusedExecutable implements PreparedFusedExecutable {
    private final FusedOperation fused;

    public Float32PreparedFusedExecutable(FusedOperation fused) {
        if (fused == null) {
            throw new IllegalArgumentException("fused cannot be null");
        }
        this.fused = fused;
    }

    @Override
    public void applyRangeScalar(List<Tensor> inputs, Tensor out, CpuKernelContext context, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        Float32FusedExecutor.applyRangeScalar(fused, inputs, out, context, options, startInclusive, endExclusive);
    }

    @Override
    public void applyRangeVector(List<Tensor> inputs, Tensor out, CpuKernelContext context, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        Float32FusedExecutor.applyRangeVector(fused, inputs, out, context, options, startInclusive, endExclusive);
    }
}
