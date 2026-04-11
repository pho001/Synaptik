package backend.kernels.cpu.nn;

import backend.kernels.cpu.*;

import operations.Operation;
import operations.avgPool2d;
import tensor.Tensor;

import java.util.List;

public final class CpuAvgPool2dKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.avgForwardF64(require(op), inputs.get(0), node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.avgForwardF32(require(op), inputs.get(0), node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        Pool2dKernelSupport.avgForwardBF16(require(op), inputs.get(0), node);
    }

    private static avgPool2d require(Operation op) {
        if (!(op instanceof avgPool2d pool)) {
            throw new IllegalArgumentException("CpuAvgPool2dKernel requires avgPool2d operation");
        }
        return pool;
    }
}
