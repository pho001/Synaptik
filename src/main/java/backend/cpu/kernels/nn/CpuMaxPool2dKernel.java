package backend.cpu.kernels.nn;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.nn.pool.maxPool2d;
import tensor.Tensor;

import java.util.List;

public final class CpuMaxPool2dKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.maxForwardF64(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.maxForwardF32(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.maxForwardBF16(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    private static maxPool2d require(Operation op) {
        if (!(op instanceof maxPool2d pool)) {
            throw new IllegalArgumentException("CpuMaxPool2dKernel requires maxPool2d operation");
        }
        return pool;
    }
}
