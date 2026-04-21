package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import operations.reduction.logSoftmaxGrad;
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
        float[] primaryContinuation = context.inputFloatContinuation(0, pair[0].getFlatDataSize());
        float[] gradContinuation = context.inputFloatContinuation(1, pair[1].getFlatDataSize());
        if (context.publishFloatContinuation() && context.cpuWorkspace() != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            SoftmaxGradExecutor.executeLogSoftmaxBF16ToFloat(
                    pair[0],
                    primaryContinuation,
                    pair[1],
                    gradContinuation,
                    out,
                    grad.getDimension(),
                    context
            );
            context.cpuWorkspace().publishFloatContinuation(node.getFlatDataSize());
            return;
        }
        if (primaryContinuation != null || gradContinuation != null) {
            SoftmaxGradExecutor.executeLogSoftmaxBF16(
                    pair[0],
                    primaryContinuation,
                    pair[1],
                    gradContinuation,
                    node,
                    grad.getDimension(),
                    context
            );
            return;
        }
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
