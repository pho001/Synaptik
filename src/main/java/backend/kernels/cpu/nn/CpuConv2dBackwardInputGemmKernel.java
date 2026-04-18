package backend.kernels.cpu.nn;

import backend.kernels.cpu.*;
import operations.Operation;
import operations.nn.conv.conv2dBackwardInputGemm;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dBackwardInputGemmKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInputGemm grad = require(op);
        Conv2dGemmBackend.backwardInputF64(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInputGemm grad = require(op);
        Conv2dGemmBackend.backwardInputF32(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardInputGemm grad = require(op);
        Conv2dGemmBackend.backwardInputBF16(grad, inputs.get(0), inputs.get(1), node, context);
    }

    private static conv2dBackwardInputGemm require(Operation op) {
        if (!(op instanceof conv2dBackwardInputGemm grad)) {
            throw new IllegalArgumentException("CpuConv2dBackwardInputGemmKernel requires conv2dBackwardInputGemm operation");
        }
        return grad;
    }
}
