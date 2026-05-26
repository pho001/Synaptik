package backend.cpu.kernels.index;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.index.scatterAxisAdd;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterAxisAddKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] triple = requireTriple(op, inputs);
        GatherAxisLoops.scatterAxisAddF64(triple[0], triple[1], triple[2], node, ((scatterAxisAdd) op).getAxis());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] triple = requireTriple(op, inputs);
        GatherAxisLoops.scatterAxisAddF32(triple[0], triple[1], triple[2], node, ((scatterAxisAdd) op).getAxis());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] triple = requireTriple(op, inputs);
        GatherAxisLoops.scatterAxisAddBF16(triple[0], triple[1], triple[2], node, ((scatterAxisAdd) op).getAxis());
    }

    private static Tensor[] requireTriple(Operation op, List<Tensor> inputs) {
        if (!(op instanceof scatterAxisAdd)) {
            throw new IllegalArgumentException("CpuScatterAxisAddKernel requires scatterAxisAdd operation.");
        }
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterAxisAdd expects exactly three inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
