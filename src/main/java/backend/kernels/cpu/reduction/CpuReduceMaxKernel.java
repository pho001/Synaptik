package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.reduction.MinMaxReduceExecutor;
import operations.Operation;
import operations.reduction.reduceMax;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceMaxKernel implements CpuKernel {
    private static final MinMaxReduceExecutor EXECUTOR = new MinMaxReduceExecutor();

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceMax reduction)) {
            throw new IllegalArgumentException("CpuReduceMaxKernel requires reduceMax operation");
        }
        EXECUTOR.execute(reduction, inputs.getFirst(), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceMax reduction)) {
            throw new IllegalArgumentException("CpuReduceMaxKernel requires reduceMax operation");
        }
        EXECUTOR.executeF32(reduction, inputs.getFirst(), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof reduceMax reduction)) {
            throw new IllegalArgumentException("CpuReduceMaxKernel requires reduceMax operation");
        }
        EXECUTOR.executeBF16(reduction, inputs.getFirst(), node, context);
    }
}
