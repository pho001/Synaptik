package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.layout.sliceScatterAdd;
import tensor.Tensor;

import java.util.List;

public final class CpuSliceScatterAddKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceScatterAdd(op, inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceScatterAdd(op, inputs, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
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
