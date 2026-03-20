package Backend.kernels.cpu;

import Operations.Operation;
import Operations.sum;
import Tensor.Tensor;
import Backend.kernels.cpu.reduction.SumExecutor;

import java.util.List;

public class CpuSumKernel implements CpuKernel {
    private static final SumExecutor EXECUTOR = new SumExecutor();

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        if (!(op instanceof sum reduction)) {
            throw new IllegalArgumentException("CpuSumKernel requires sum operation");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Sum expects exactly one input tensor");
        }
        EXECUTOR.execute(reduction, inputs.getFirst(), node, config);
    }
}
