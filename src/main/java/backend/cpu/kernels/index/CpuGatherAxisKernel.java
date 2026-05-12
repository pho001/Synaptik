package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.index.gatherAxis;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherAxisKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisF64(pair[0], pair[1], node, ((gatherAxis) op).getAxis(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisF32(pair[0], pair[1], node, ((gatherAxis) op).getAxis(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisBF16(pair[0], pair[1], node, ((gatherAxis) op).getAxis(), context);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisBOOL(pair[0], pair[1], node, ((gatherAxis) op).getAxis(), context);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        IndexExecutor.gatherAxisI32(pair[0], pair[1], node, ((gatherAxis) op).getAxis(), context);
    }

    private static Tensor[] requirePair(Operation op, List<Tensor> inputs) {
        if (!(op instanceof gatherAxis)) {
            throw new IllegalArgumentException("CpuGatherAxisKernel requires gatherAxis operation.");
        }
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherAxis expects exactly two inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
