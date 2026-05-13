package backend.cpu.kernels.index;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.index.gather;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        IndexExecutor.gatherF64(pair[0], pair[1], node, gatherOp.getDimension(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        IndexExecutor.gatherF32(pair[0], pair[1], node, gatherOp.getDimension(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        IndexExecutor.gatherBF16(pair[0], pair[1], node, gatherOp.getDimension(), context);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        IndexExecutor.gatherBOOL(pair[0], pair[1], node, gatherOp.getDimension(), context);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        IndexExecutor.gatherI32(pair[0], pair[1], node, gatherOp.getDimension(), context);
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        IndexExecutor.gatherI64(pair[0], pair[1], node, gatherOp.getDimension(), context);
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Gather expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
