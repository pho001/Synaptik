package graph.fused.vector;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.fused.Float32FusedExecutor;
import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.fused.PreparedFusedExecutable;
import operations.FusedOperation;
import tensor.Tensor;

import java.util.List;

public final class Float32PreparedFusedExecutable implements PreparedFusedExecutable {
    private final DirectLinearF32Program program;

    public Float32PreparedFusedExecutable(FusedOperation fused, DirectLinearF32Program program) {
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
