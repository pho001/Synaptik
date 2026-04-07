package graph.fused.vector;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.Float64FusedExecutor;
import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.fused.PreparedFusedExecutable;
import operations.FusedOperation;
import tensor.Tensor;

import java.util.List;

public final class Float64PreparedFusedExecutable implements PreparedFusedExecutable {
    private final FusedOperation fused;

    public Float64PreparedFusedExecutable(FusedOperation fused) {
        if (fused == null) {
            throw new IllegalArgumentException("fused cannot be null");
        }
        this.fused = fused;
    }

    @Override
    public void applyRangeScalar(List<Tensor> inputs, Tensor out, CpuKernelContext context, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        Float64FusedExecutor.applyRangeScalar(fused, inputs, out, context, options, startInclusive, endExclusive);
    }

    @Override
    public void applyRangeVector(List<Tensor> inputs, Tensor out, CpuKernelContext context, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        Float64FusedExecutor.applyRangeVector(fused, inputs, out, context, options, startInclusive, endExclusive);
    }
}
