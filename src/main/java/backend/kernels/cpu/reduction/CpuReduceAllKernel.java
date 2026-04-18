package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.reduction.BoolReduceExecutor;
import operations.Operation;
import operations.reduction.reduceAll;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceAllKernel implements CpuKernel {
    private static final BoolReduceExecutor EXECUTOR = new BoolReduceExecutor();

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceAll reduction)) {
            throw new IllegalArgumentException("CpuReduceAllKernel requires reduceAll operation");
        }
        EXECUTOR.execute(reduction, inputs.getFirst(), node, context);
    }
}
