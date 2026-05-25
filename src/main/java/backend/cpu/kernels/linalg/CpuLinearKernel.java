package backend.cpu.kernels.linalg;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.linalg.linear;
import tensor.Tensor;

import java.util.List;

public final class CpuLinearKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        linear linear = require(op);
        LinearExecutor.forwardF64(linear, inputs.get(0), inputs.get(1), linear.hasBias() ? inputs.get(2) : null, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        linear linear = require(op);
        LinearExecutor.forwardF32(linear, inputs.get(0), inputs.get(1), linear.hasBias() ? inputs.get(2) : null, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        linear linear = require(op);
        LinearExecutor.forwardBF16(linear, inputs.get(0), inputs.get(1), linear.hasBias() ? inputs.get(2) : null, node, context);
    }

    private static linear require(Operation op) {
        if (!(op instanceof linear linear)) {
            throw new IllegalArgumentException("CpuLinearKernel requires linear operation");
        }
        return linear;
    }
}
