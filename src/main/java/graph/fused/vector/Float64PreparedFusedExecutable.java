package graph.fused.vector;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.fused.Float64FusedExecutor;
import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.fused.PreparedFusedExecutable;
import operations.FusedOperation;
import tensor.Tensor;

import java.util.List;

public final class Float64PreparedFusedExecutable implements PreparedFusedExecutable {
    private final DirectLinearF64Program program;

    public Float64PreparedFusedExecutable(FusedOperation fused, DirectLinearF64Program program) {
        if (fused == null || program == null) {
            throw new IllegalArgumentException("fused/program cannot be null");
        }
        this.program = program;
    }

    @Override
    public void applyRangeScalar(List<Tensor> inputs, Tensor out, CpuKernelContext context, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        program.applyScalar(inputs, out, startInclusive, endExclusive, options);
    }

    @Override
    public void applyRangeVector(List<Tensor> inputs, Tensor out, CpuKernelContext context, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        program.applyVector(inputs, out, startInclusive, endExclusive);
    }
}
