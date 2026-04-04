package backend.kernels.cpu;

import operations.Operation;
import operations.maxPool2d;
import tensor.Tensor;

import java.util.List;

public final class CpuMaxPool2dKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.maxForwardF64(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.maxForwardF32(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.maxForwardF16(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    private static maxPool2d require(Operation op) {
        if (!(op instanceof maxPool2d pool)) {
            throw new IllegalArgumentException("CpuMaxPool2dKernel requires maxPool2d operation");
        }
        return pool;
    }
}
