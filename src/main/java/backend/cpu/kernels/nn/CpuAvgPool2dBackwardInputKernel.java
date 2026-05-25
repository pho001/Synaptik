package backend.cpu.kernels.nn;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.nn.pool.avgPool2dBackwardInput;
import tensor.Tensor;

import java.util.List;

public final class CpuAvgPool2dBackwardInputKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgBackwardInputF64(require(op), inputs.get(0), node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgBackwardInputF32(require(op), inputs.get(0), node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgBackwardInputBF16(require(op), inputs.get(0), node);
    }

    private static avgPool2dBackwardInput require(Operation op) {
        if (!(op instanceof avgPool2dBackwardInput pool)) {
            throw new IllegalArgumentException("CpuAvgPool2dBackwardInputKernel requires avgPool2dBackwardInput operation");
        }
        return pool;
    }
}
