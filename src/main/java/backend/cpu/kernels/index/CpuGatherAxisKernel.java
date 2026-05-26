package backend.cpu.kernels.index;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.index.gatherAxis;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherAxisKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherAxisLoops.gatherAxisF64(pair[0], pair[1], node, ((gatherAxis) op).getAxis());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherAxisLoops.gatherAxisF32(pair[0], pair[1], node, ((gatherAxis) op).getAxis());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherAxisLoops.gatherAxisBF16(pair[0], pair[1], node, ((gatherAxis) op).getAxis());
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherAxisLoops.gatherAxisBOOL(pair[0], pair[1], node, ((gatherAxis) op).getAxis());
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherAxisLoops.gatherAxisI32(pair[0], pair[1], node, ((gatherAxis) op).getAxis());
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(op, inputs);
        GatherAxisLoops.gatherAxisI64(pair[0], pair[1], node, ((gatherAxis) op).getAxis());
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
