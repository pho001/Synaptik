package backend.cpu.kernels.nn;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.nn.conv.conv2d;
import tensor.Tensor;

import java.util.List;

public final class CpuConv2dKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2d conv = require(op);
        Conv2dExecutor.forwardF64(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2d conv = require(op);
        Conv2dExecutor.forwardF32(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        conv2d conv = require(op);
        Conv2dExecutor.forwardBF16(conv, inputs.get(0), inputs.get(1), inputs.size() > 2 ? inputs.get(2) : null, node, context);
    }

    private static conv2d require(Operation op) {
        if (!(op instanceof conv2d conv)) {
            throw new IllegalArgumentException("CpuConv2dKernel requires conv2d operation");
        }
        return conv;
    }
}
