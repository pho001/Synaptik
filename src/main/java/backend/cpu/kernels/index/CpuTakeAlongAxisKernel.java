package backend.cpu.kernels.index;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.index.takeAlongAxis;
import tensor.Tensor;

import java.util.List;

public final class CpuTakeAlongAxisKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisF64(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisF32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisBF16(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisBOOL(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisI32(pair[0], pair[1], node, gatherOp.getDimension());
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisI64(pair[0], pair[1], node, gatherOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxis expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
