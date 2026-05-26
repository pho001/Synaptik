package backend.cpu.kernels.index;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.index.gatherNd;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherNdKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherNdLoops.gatherNdF64(pair[0], pair[1], node, requireOp(op).getBatchDims());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherNdLoops.gatherNdF32(pair[0], pair[1], node, requireOp(op).getBatchDims());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherNdLoops.gatherNdBF16(pair[0], pair[1], node, requireOp(op).getBatchDims());
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherNdLoops.gatherNdBOOL(pair[0], pair[1], node, requireOp(op).getBatchDims());
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherNdLoops.gatherNdI32(pair[0], pair[1], node, requireOp(op).getBatchDims());
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherNdLoops.gatherNdI64(pair[0], pair[1], node, requireOp(op).getBatchDims());
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
