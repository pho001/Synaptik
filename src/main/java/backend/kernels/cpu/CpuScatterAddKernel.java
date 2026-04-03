package backend.kernels.cpu;

import operations.Operation;
import operations.scatterAdd;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterAddKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof scatterAdd scatterAddOp)) {
            throw new IllegalArgumentException("CpuScatterAddKernel requires scatterAdd operation");
        }
        Tensor[] triple = requireTriple(inputs);
        GatherSupport.scatterAddF64(triple[0], triple[1], triple[2], node, scatterAddOp.getDimension());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof scatterAdd scatterAddOp)) {
            throw new IllegalArgumentException("CpuScatterAddKernel requires scatterAdd operation");
        }
        Tensor[] triple = requireTriple(inputs);
        GatherSupport.scatterAddF32(triple[0], triple[1], triple[2], node, scatterAddOp.getDimension());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof scatterAdd scatterAddOp)) {
            throw new IllegalArgumentException("CpuScatterAddKernel requires scatterAdd operation");
        }
        Tensor[] triple = requireTriple(inputs);
        GatherSupport.scatterAddF16(triple[0], triple[1], triple[2], node, scatterAddOp.getDimension());
    }

    private static Tensor[] requireTriple(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterAdd expects exactly three inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
