package backend.cpu.kernels.index;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.index.scatterElements;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterElementsKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterElements scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterElementsF64(triple[0], triple[1], triple[2], node, scatterOp.getAxis(), scatterOp.getReduction(), context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterElements scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterElementsF32(triple[0], triple[1], triple[2], node, scatterOp.getAxis(), scatterOp.getReduction(), context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterElements scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterElementsBF16(triple[0], triple[1], triple[2], node, scatterOp.getAxis(), scatterOp.getReduction(), context);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterElements scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterElementsBOOL(triple[0], triple[1], triple[2], node, scatterOp.getAxis(), scatterOp.getReduction(), context);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterElements scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterElementsI32(triple[0], triple[1], triple[2], node, scatterOp.getAxis(), scatterOp.getReduction(), context);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterElements scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterElementsI64(triple[0], triple[1], triple[2], node, scatterOp.getAxis(), scatterOp.getReduction(), context);
    }

    private static scatterElements requireOp(Operation op) {
        if (!(op instanceof scatterElements scatterOp)) {
            throw new IllegalArgumentException("CpuScatterElementsKernel requires scatterElements operation");
        }
        return scatterOp;
    }

    private static Tensor[] requireTriple(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterElements expects exactly three inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
