package backend.cpu.kernels.nn;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.nn.conv.conv2dBackwardInput;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dBackwardInputKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInput grad = require(op);
        Conv2dExecutor.backwardInputF64(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInput grad = require(op);
        Conv2dExecutor.backwardInputF32(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInput grad = require(op);
        Conv2dExecutor.backwardInputF16(grad, inputs.get(0), inputs.get(1), node, context);
    }

    private static conv2dBackwardInput require(Operation op) {
        if (!(op instanceof conv2dBackwardInput grad)) {
            throw new IllegalArgumentException("CpuConv2dBackwardInputKernel requires conv2dBackwardInput operation");
        }
        return grad;
    }
}
