package backend.kernels.cpu;

import operations.Operation;
import operations.gather;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.runF64(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.runF32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.runF16(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.runBOOL(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.runI32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Gather expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
