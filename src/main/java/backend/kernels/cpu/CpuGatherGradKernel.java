package backend.kernels.cpu;

import operations.Operation;
import operations.gatherGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gatherGrad gatherGradOp)) {
            throw new IllegalArgumentException("CpuGatherGradKernel requires gatherGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.scatterF64(pair[0], pair[1], node, gatherGradOp.getDimension());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gatherGrad gatherGradOp)) {
            throw new IllegalArgumentException("CpuGatherGradKernel requires gatherGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.scatterF32(pair[0], pair[1], node, gatherGradOp.getDimension());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gatherGrad gatherGradOp)) {
            throw new IllegalArgumentException("CpuGatherGradKernel requires gatherGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherSupport.scatterBF16(pair[0], pair[1], node, gatherGradOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("GatherGrad expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
