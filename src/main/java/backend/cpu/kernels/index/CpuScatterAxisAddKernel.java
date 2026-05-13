package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.index.scatterAxisAdd;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterAxisAddKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] triple = requireTriple(op, inputs);
        IndexExecutor.scatterAxisAddF64(triple[0], triple[1], triple[2], node, ((scatterAxisAdd) op).getAxis(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] triple = requireTriple(op, inputs);
        IndexExecutor.scatterAxisAddF32(triple[0], triple[1], triple[2], node, ((scatterAxisAdd) op).getAxis(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] triple = requireTriple(op, inputs);
        IndexExecutor.scatterAxisAddBF16(triple[0], triple[1], triple[2], node, ((scatterAxisAdd) op).getAxis(), context);
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
