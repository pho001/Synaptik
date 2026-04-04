package backend.kernels.cpu;

import operations.Operation;
import operations.linear;
import tensor.Tensor;

import java.util.List;

public final class CpuLinearKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        linear linear = require(op);
        LinearKernelSupport.forwardF64(linear, inputs.get(0), inputs.get(1), linear.hasBias() ? inputs.get(2) : null, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        linear linear = require(op);
        LinearKernelSupport.forwardF32(linear, inputs.get(0), inputs.get(1), linear.hasBias() ? inputs.get(2) : null, node, context);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        linear linear = require(op);
        LinearKernelSupport.forwardF16(linear, inputs.get(0), inputs.get(1), linear.hasBias() ? inputs.get(2) : null, node, context);
    }

    private static linear require(Operation op) {
        if (!(op instanceof linear linear)) {
            throw new IllegalArgumentException("CpuLinearKernel requires linear operation");
        }
        return linear;
    }
}
