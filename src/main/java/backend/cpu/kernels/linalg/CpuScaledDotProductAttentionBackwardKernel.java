package backend.cpu.kernels.linalg;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCostClass;
import operations.Operation;
import operations.linalg.scaledDotProductAttentionBackward;
import tensor.Tensor;

import java.util.List;

public final class CpuScaledDotProductAttentionBackwardKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        ScaledDotProductAttentionExecutor.executeBackwardF64(require(op).getOutputKind(), pair[0], pair[1], node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        ScaledDotProductAttentionExecutor.executeBackwardF32(require(op).getOutputKind(), pair[0], pair[1], node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        ScaledDotProductAttentionExecutor.executeBackwardBF16(require(op).getOutputKind(), pair[0], pair[1], node, context);
    }

    private static scaledDotProductAttentionBackward require(Operation op) {
        if (!(op instanceof scaledDotProductAttentionBackward backward)) {
            throw new IllegalArgumentException("CpuScaledDotProductAttentionBackwardKernel requires scaledDotProductAttentionBackward operation");
        }
        return backward;
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("scaledDotProductAttentionBackward expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }

    @Override
    public CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.HIGH;
    }
}
