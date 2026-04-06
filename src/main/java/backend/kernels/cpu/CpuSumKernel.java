package backend.kernels.cpu;

import backend.kernels.cpu.reduction.SumExecutor;
import operations.Operation;
import operations.sum;
import tensor.Tensor;

import java.util.List;

public class CpuSumKernel implements CpuKernel {
    private static final SumExecutor EXECUTOR = new SumExecutor();

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof sum reduction)) {
            throw new IllegalArgumentException("CpuSumKernel requires sum operation");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Sum expects exactly one input tensor");
        }
        EXECUTOR.execute(reduction, inputs.getFirst(), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof sum reduction)) {
            throw new IllegalArgumentException("CpuSumKernel requires sum operation");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Sum expects exactly one input tensor");
        }
        EXECUTOR.executeF32(reduction, inputs.getFirst(), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof sum reduction)) {
            throw new IllegalArgumentException("CpuSumKernel requires sum operation");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Sum expects exactly one input tensor");
        }
        EXECUTOR.executeBF16(reduction, inputs.getFirst(), node, context);
    }
}
