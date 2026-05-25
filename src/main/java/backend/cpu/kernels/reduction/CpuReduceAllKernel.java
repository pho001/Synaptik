package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import backend.cpu.kernels.reduction.BoolReduceExecutor;
import operations.Operation;
import operations.reduction.reduceAll;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceAllKernel extends TypedCpuKernel {
    private static final BoolReduceExecutor EXECUTOR = new BoolReduceExecutor();

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceAll reduction)) {
            throw new IllegalArgumentException("CpuReduceAllKernel requires reduceAll operation");
        }
        EXECUTOR.execute(reduction, inputs.getFirst(), node, context);
    }
}
