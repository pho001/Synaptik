package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.layout.sliceGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuSliceGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceGrad(op, inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceGrad(op, inputs, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceGrad(op, inputs, node);
    }

    private static void sliceGrad(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof sliceGrad sliceGradOp)) {
            throw new IllegalArgumentException("CpuSliceGradKernel requires sliceGrad operation.");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("sliceGrad expects exactly one input.");
        }
        LayoutGradExecutor.sliceGrad(sliceGradOp, inputs.getFirst(), node);
    }
}
