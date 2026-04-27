package backend.cpu.kernels.reduction;

import backend.cpu.kernels.*;

import backend.cpu.kernels.reduction.MinMaxReduceExecutor;
import operations.Operation;
import operations.reduction.reduceMin;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceMinKernel implements CpuKernel {
    private static final MinMaxReduceExecutor EXECUTOR = new MinMaxReduceExecutor();

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceMin reduction)) {
            throw new IllegalArgumentException("CpuReduceMinKernel requires reduceMin operation");
        }
        EXECUTOR.execute(reduction, inputs.getFirst(), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceMin reduction)) {
            throw new IllegalArgumentException("CpuReduceMinKernel requires reduceMin operation");
        }
        EXECUTOR.executeF32(reduction, inputs.getFirst(), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceMin reduction)) {
            throw new IllegalArgumentException("CpuReduceMinKernel requires reduceMin operation");
        }
        EXECUTOR.executeBF16(reduction, inputs.getFirst(), node, context);
    }
}
