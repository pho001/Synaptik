package backend.kernels.cpu.nn;

import backend.kernels.cpu.*;
import operations.Operation;
import operations.nn.conv.conv2dBackwardWeightGemm;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dBackwardWeightGemmKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeightGemm grad = require(op);
        Conv2dGemmBackend.backwardWeightF64(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeightGemm grad = require(op);
        Conv2dGemmBackend.backwardWeightF32(grad, inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dBackwardWeightGemm grad = require(op);
        Conv2dGemmBackend.backwardWeightBF16(grad, inputs.get(0), inputs.get(1), node, context);
    }

    private static conv2dBackwardWeightGemm require(Operation op) {
        if (!(op instanceof conv2dBackwardWeightGemm grad)) {
            throw new IllegalArgumentException("CpuConv2dBackwardWeightGemmKernel requires conv2dBackwardWeightGemm operation");
        }
        return grad;
    }
}
