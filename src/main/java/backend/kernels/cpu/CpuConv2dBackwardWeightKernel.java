package backend.kernels.cpu;

import operations.Operation;
import operations.conv2dBackwardWeight;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dBackwardWeightKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeight grad = require(op);
        Conv2dKernelSupport.backwardWeightF64(grad, inputs.get(0), inputs.get(1), node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeight grad = require(op);
        Conv2dKernelSupport.backwardWeightF32(grad, inputs.get(0), inputs.get(1), node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeight grad = require(op);
        Conv2dKernelSupport.backwardWeightF16(grad, inputs.get(0), inputs.get(1), node);
    }

    private static conv2dBackwardWeight require(Operation op) {
        if (!(op instanceof conv2dBackwardWeight grad)) {
            throw new IllegalArgumentException("CpuConv2dBackwardWeightKernel requires conv2dBackwardWeight operation");
        }
        return grad;
    }
}
