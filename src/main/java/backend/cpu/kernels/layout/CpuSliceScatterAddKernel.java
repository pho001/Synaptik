package backend.cpu.kernels.layout;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.layout.sliceScatterAdd;
import tensor.Tensor;

import java.util.List;

public final class CpuSliceScatterAddKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceScatterAdd(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceScatterAdd(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceScatterAdd(op, inputs, node);
    }

    private static void sliceScatterAdd(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof sliceScatterAdd sliceOp)) {
            throw new IllegalArgumentException("CpuSliceScatterAddKernel requires sliceScatterAdd operation.");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("sliceScatterAdd expects exactly one input.");
        }
        LayoutGradExecutor.sliceScatterAdd(sliceOp, inputs.getFirst(), node);
    }
}
