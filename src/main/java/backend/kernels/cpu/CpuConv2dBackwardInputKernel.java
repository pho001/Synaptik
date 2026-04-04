package backend.kernels.cpu;

import operations.Operation;
import operations.conv2dBackwardInput;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dBackwardInputKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInput grad = require(op);
        Conv2dKernelSupport.backwardInputF64(grad, inputs.get(0), inputs.get(1), node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInput grad = require(op);
        Conv2dKernelSupport.backwardInputF32(grad, inputs.get(0), inputs.get(1), node);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInput grad = require(op);
        Conv2dKernelSupport.backwardInputF16(grad, inputs.get(0), inputs.get(1), node);
    }

    private static conv2dBackwardInput require(Operation op) {
        if (!(op instanceof conv2dBackwardInput grad)) {
            throw new IllegalArgumentException("CpuConv2dBackwardInputKernel requires conv2dBackwardInput operation");
        }
        return grad;
    }
}
