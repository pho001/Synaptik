package backend.kernels.cpu.index;

import backend.kernels.cpu.*;

import operations.Operation;
import operations.takeAlongAxis;
import tensor.Tensor;

import java.util.List;

public final class CpuTakeAlongAxisKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisF64(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisF32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisF16(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisBOOL(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisI32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxis expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
