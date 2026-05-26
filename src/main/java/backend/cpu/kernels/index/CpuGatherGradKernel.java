package backend.cpu.kernels.index;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.index.gatherGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherGradKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gatherGrad gatherGradOp)) {
            throw new IllegalArgumentException("CpuGatherGradKernel requires gatherGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherGradF64(pair[0], pair[1], node, gatherGradOp.getDimension());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gatherGrad gatherGradOp)) {
            throw new IllegalArgumentException("CpuGatherGradKernel requires gatherGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherGradF32(pair[0], pair[1], node, gatherGradOp.getDimension());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof gatherGrad gatherGradOp)) {
            throw new IllegalArgumentException("CpuGatherGradKernel requires gatherGrad operation");
        }
        Tensor[] pair = requirePair(inputs);
        GatherLoops.gatherGradBF16(pair[0], pair[1], node, gatherGradOp.getDimension());
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("GatherGrad expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
