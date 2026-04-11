package backend.kernels.cpu.nn;

import backend.kernels.cpu.*;

import operations.Operation;
import operations.maxPool2dBackwardInput;
import tensor.Tensor;

import java.util.List;

public final class CpuMaxPool2dBackwardInputKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.maxBackwardInputF64(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.maxBackwardInputF32(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.maxBackwardInputBF16(require(op), inputs.get(0), node, context.cpuWorkspace().requireIntWorkspace());
    }

    private static maxPool2dBackwardInput require(Operation op) {
        if (!(op instanceof maxPool2dBackwardInput pool)) {
            throw new IllegalArgumentException("CpuMaxPool2dBackwardInputKernel requires maxPool2dBackwardInput operation");
        }
        return pool;
    }
}
