package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.index.gatherNd;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherNdKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdF64(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdF32(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdBF16(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdBOOL(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdI32(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherNdI64(pair[0], pair[1], node, requireOp(op).getBatchDims(), context);
    }

    private static gatherNd requireOp(Operation op) {
        if (!(op instanceof gatherNd gatherOp)) {
            throw new IllegalArgumentException("CpuGatherNdKernel requires gatherNd operation.");
        }
        return gatherOp;
    }

    private static Tensor[] requirePair(Operation op, List<Tensor> inputs) {
        requireOp(op);
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherNd expects exactly two inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
