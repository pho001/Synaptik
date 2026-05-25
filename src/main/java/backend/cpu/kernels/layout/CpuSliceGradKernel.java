package backend.cpu.kernels.layout;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.layout.sliceGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuSliceGradKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceGrad(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        sliceGrad(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
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
