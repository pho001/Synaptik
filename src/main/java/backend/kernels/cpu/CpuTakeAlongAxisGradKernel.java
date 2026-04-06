package backend.kernels.cpu;

import operations.Operation;
import operations.takeAlongAxisGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuTakeAlongAxisGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxisGrad gradOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisGradKernel requires takeAlongAxisGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisScatterF64(pair[0], pair[1], node, gradOp.getDimension());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxisGrad gradOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisGradKernel requires takeAlongAxisGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisScatterF32(pair[0], pair[1], node, gradOp.getDimension());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof takeAlongAxisGrad gradOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisGradKernel requires takeAlongAxisGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.takeAlongAxisScatterBF16(pair[0], pair[1], node, gradOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxisGrad expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
