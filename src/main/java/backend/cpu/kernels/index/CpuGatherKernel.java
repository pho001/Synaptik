package backend.cpu.kernels.index;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.index.gather;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherF64(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherF32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherBF16(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherBOOL(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherI32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherI64(pair[0], pair[1], node, gatherOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Gather expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
