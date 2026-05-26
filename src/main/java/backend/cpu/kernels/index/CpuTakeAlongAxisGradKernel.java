package backend.cpu.kernels.index;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.index.takeAlongAxisGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuTakeAlongAxisGradKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxisGrad gradOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisGradKernel requires takeAlongAxisGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisGradF64(pair[0], pair[1], node, gradOp.getDimension());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxisGrad gradOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisGradKernel requires takeAlongAxisGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisGradF32(pair[0], pair[1], node, gradOp.getDimension());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxisGrad gradOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisGradKernel requires takeAlongAxisGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        TakeAlongAxisLoops.takeAlongAxisGradBF16(pair[0], pair[1], node, gradOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxisGrad expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
