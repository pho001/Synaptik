package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import operations.logSoftmaxGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuLogSoftmaxGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        logSoftmaxGrad grad = require(op);
        Tensor[] pair = requirePair(inputs);
        SoftmaxGradExecutor.executeLogSoftmaxF64(pair[0], pair[1], node, grad.getDimension(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        logSoftmaxGrad grad = require(op);
        Tensor[] pair = requirePair(inputs);
        SoftmaxGradExecutor.executeLogSoftmaxF32(pair[0], pair[1], node, grad.getDimension(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        logSoftmaxGrad grad = require(op);
        Tensor[] pair = requirePair(inputs);
        SoftmaxGradExecutor.executeLogSoftmaxBF16(pair[0], pair[1], node, grad.getDimension(), context);
    }

    private static logSoftmaxGrad require(Operation op) {
        if (!(op instanceof logSoftmaxGrad grad)) {
            throw new IllegalArgumentException("CpuLogSoftmaxGradKernel requires logSoftmaxGrad operation");
        }
        return grad;
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("logSoftmaxGrad expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
