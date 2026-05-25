package backend.cpu.kernels.index;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.index.scatterNd;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterNdKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterNd scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterNdF64(triple[0], triple[1], triple[2], node, scatterOp.getReduction(), scatterOp.getBatchDims(), context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterNd scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterNdF32(triple[0], triple[1], triple[2], node, scatterOp.getReduction(), scatterOp.getBatchDims(), context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterNd scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterNdBF16(triple[0], triple[1], triple[2], node, scatterOp.getReduction(), scatterOp.getBatchDims(), context);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterNd scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterNdBOOL(triple[0], triple[1], triple[2], node, scatterOp.getReduction(), scatterOp.getBatchDims(), context);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterNd scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterNdI32(triple[0], triple[1], triple[2], node, scatterOp.getReduction(), scatterOp.getBatchDims(), context);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scatterNd scatterOp = requireOp(op);
        Tensor[] triple = requireTriple(inputs);
        IndexExecutor.scatterNdI64(triple[0], triple[1], triple[2], node, scatterOp.getReduction(), scatterOp.getBatchDims(), context);
    }

    private static scatterNd requireOp(Operation op) {
        if (!(op instanceof scatterNd scatterOp)) {
            throw new IllegalArgumentException("CpuScatterNdKernel requires scatterNd operation");
        }
        return scatterOp;
    }

    private static Tensor[] requireTriple(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterNd expects exactly three inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
