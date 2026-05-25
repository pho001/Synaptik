package backend.cpu.kernels.linalg;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.plan.CpuKernelCostClass;
import operations.Operation;
import operations.linalg.scaledDotProductAttentionBackward;
import tensor.Tensor;

import java.util.List;

public final class CpuScaledDotProductAttentionBackwardKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        ScaledDotProductAttentionExecutor.executeBackwardF64(require(op).getOutputKind(), pair[0], pair[1], node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        ScaledDotProductAttentionExecutor.executeBackwardF32(require(op).getOutputKind(), pair[0], pair[1], node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
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
