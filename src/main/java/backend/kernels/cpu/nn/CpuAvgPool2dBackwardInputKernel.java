package backend.kernels.cpu.nn;

import backend.kernels.cpu.*;

import operations.Operation;
import operations.avgPool2dBackwardInput;
import tensor.Tensor;

import java.util.List;

public final class CpuAvgPool2dBackwardInputKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgBackwardInputF64(require(op), inputs.get(0), node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgBackwardInputF32(require(op), inputs.get(0), node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dExecutor.avgBackwardInputBF16(require(op), inputs.get(0), node);
    }

    private static avgPool2dBackwardInput require(Operation op) {
        if (!(op instanceof avgPool2dBackwardInput pool)) {
            throw new IllegalArgumentException("CpuAvgPool2dBackwardInputKernel requires avgPool2dBackwardInput operation");
        }
        return pool;
    }
}
