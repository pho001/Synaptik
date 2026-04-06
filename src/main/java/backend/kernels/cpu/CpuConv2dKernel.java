package backend.kernels.cpu;

import operations.Operation;
import operations.conv2d;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2d conv = require(op);
        Conv2dKernelSupport.forwardF64(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2d conv = require(op);
        Conv2dKernelSupport.forwardF32(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2d conv = require(op);
        Conv2dKernelSupport.forwardBF16(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node);
    }

    private static conv2d require(Operation op) {
        if (!(op instanceof conv2d conv)) {
            throw new IllegalArgumentException("CpuConv2dKernel requires conv2d operation");
        }
        return conv;
    }
}
