package backend.cpu.kernels.nn;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.nn.pool.maxPool2dBackwardInput;
import tensor.Tensor;

import java.util.List;

public final class CpuMaxPool2dBackwardInputKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        Pool2dExecutor.maxBackwardInputF64(require(op), pair[0], pair[1], node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        Pool2dExecutor.maxBackwardInputF32(require(op), pair[0], pair[1], node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Tensor[] pair = requirePair(inputs);
        Pool2dExecutor.maxBackwardInputBF16(require(op), pair[0], pair[1], node);
    }

    private static maxPool2dBackwardInput require(Operation op) {
        if (!(op instanceof maxPool2dBackwardInput pool)) {
            throw new IllegalArgumentException("CpuMaxPool2dBackwardInputKernel requires maxPool2dBackwardInput operation");
        }
        return pool;
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("maxPool2d backward expects output gradient and original source input");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
