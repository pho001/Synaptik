package backend.kernels.cpu.nn;

import backend.kernels.cpu.*;

import operations.Operation;
import operations.conv2dGemm;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dGemmKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dGemm conv = require(op);
        Conv2dGemmKernelSupport.forwardF64(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dGemm conv = require(op);
        Conv2dGemmKernelSupport.forwardF32(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2dGemm conv = require(op);
        Conv2dGemmKernelSupport.forwardBF16(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node, context);
    }

    private static conv2dGemm require(Operation op) {
        if (!(op instanceof conv2dGemm conv)) {
            throw new IllegalArgumentException("CpuConv2dGemmKernel requires conv2dGemm operation");
        }
        return conv;
    }
}
