package backend.cpu.kernels.index;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.index.gatherAxisGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherAxisGradKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisGradF64(pair[0], pair[1], node, ((gatherAxisGrad) op).getAxis(), context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisGradF32(pair[0], pair[1], node, ((gatherAxisGrad) op).getAxis(), context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisGradBF16(pair[0], pair[1], node, ((gatherAxisGrad) op).getAxis(), context);
    }

    private static Tensor[] requirePair(Operation op, List<Tensor> inputs) {
        if (!(op instanceof gatherAxisGrad)) {
            throw new IllegalArgumentException("CpuGatherAxisGradKernel requires gatherAxisGrad operation.");
        }
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherAxisGrad expects exactly two inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
