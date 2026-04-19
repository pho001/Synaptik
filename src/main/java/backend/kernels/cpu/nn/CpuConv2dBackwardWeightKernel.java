package backend.kernels.cpu.nn;

import backend.kernels.cpu.*;

import operations.Operation;
import operations.nn.conv.conv2dBackwardWeight;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dBackwardWeightKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeight grad = require(op);
        Conv2dExecutor.backwardWeightF64(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeight grad = require(op);
        Conv2dExecutor.backwardWeightF32(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeight grad = require(op);
        Conv2dExecutor.backwardWeightF16(grad, inputs.get(0), inputs.get(1), node, context);
    }

    private static conv2dBackwardWeight require(Operation op) {
        if (!(op instanceof conv2dBackwardWeight grad)) {
            throw new IllegalArgumentException("CpuConv2dBackwardWeightKernel requires conv2dBackwardWeight operation");
        }
        return grad;
    }
}
