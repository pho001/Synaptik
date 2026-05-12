package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.index.gatherNdGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherNdGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdGradF64(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdGradF32(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdGradBF16(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    private static gatherNdGrad requireOp(Operation op) {
        if (!(op instanceof gatherNdGrad gradOp)) {
            throw new IllegalArgumentException("CpuGatherNdGradKernel requires gatherNdGrad operation.");
        }
        return gradOp;
    }

    private static Tensor[] requirePair(Operation op, List<Tensor> inputs) {
        requireOp(op);
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherNdGrad expects exactly two inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
