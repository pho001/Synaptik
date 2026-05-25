package backend.cpu.kernels.nn;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.nn.pool.avgPool2d;
import tensor.Tensor;

import java.util.List;

public final class CpuAvgPool2dKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgForwardF64(require(op), inputs.get(0), node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgForwardF32(require(op), inputs.get(0), node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgForwardBF16(require(op), inputs.get(0), node);
    }

    private static avgPool2d require(Operation op) {
        if (!(op instanceof avgPool2d pool)) {
            throw new IllegalArgumentException("CpuAvgPool2dKernel requires avgPool2d operation");
        }
        return pool;
    }
}
