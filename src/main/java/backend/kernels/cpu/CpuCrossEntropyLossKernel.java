package backend.kernels.cpu;

import backend.kernels.cpu.reduction.CrossEntropyLossExecutor;
import operations.Operation;
import operations.crossEntropyLoss;
import tensor.Tensor;

import java.util.List;

public final class CpuCrossEntropyLossKernel implements CpuKernel {
    private static final CrossEntropyLossExecutor EXECUTOR = new CrossEntropyLossExecutor();

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof crossEntropyLoss loss)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossKernel requires crossEntropyLoss operation");
        }
        Tensor[] pair = requirePair(inputs);
        EXECUTOR.execute(loss, pair[0], pair[1], node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof crossEntropyLoss loss)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossKernel requires crossEntropyLoss operation");
        }
        Tensor[] pair = requirePair(inputs);
        EXECUTOR.executeF32(loss, pair[0], pair[1], node, context);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof crossEntropyLoss loss)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossKernel requires crossEntropyLoss operation");
        }
        Tensor[] pair = requirePair(inputs);
        EXECUTOR.executeF16(loss, pair[0], pair[1], node, context);
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Cross entropy loss expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
