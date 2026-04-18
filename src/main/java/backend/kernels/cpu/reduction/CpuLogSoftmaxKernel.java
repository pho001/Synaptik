package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import operations.reduction.logSoftmax;
import tensor.Tensor;

import java.util.List;

public final class CpuLogSoftmaxKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        logSoftmax reduction = require(op);
        Tensor input = CpuSoftmaxKernel.requireSingleInput(inputs);
        SoftmaxLikeExecutor.executeF64(SoftmaxLikeReduction.LOG_SOFTMAX, input, node, reduction.getDimension(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        logSoftmax reduction = require(op);
        Tensor input = CpuSoftmaxKernel.requireSingleInput(inputs);
        SoftmaxLikeExecutor.executeF32(SoftmaxLikeReduction.LOG_SOFTMAX, input, node, reduction.getDimension(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        logSoftmax reduction = require(op);
        Tensor input = CpuSoftmaxKernel.requireSingleInput(inputs);
        float[] continuation = context.inputFloatContinuation(0, input.getFlatDataSize());
        if (context.publishFloatContinuation() && context.cpuWorkspace() != null && continuation != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            SoftmaxLikeExecutor.executeF32ToFloat(SoftmaxLikeReduction.LOG_SOFTMAX, input, continuation, out, reduction.getDimension(), context);
            context.cpuWorkspace().publishFloatContinuation(input.getFlatDataSize());
            return;
        }
        if (continuation != null) {
            SoftmaxLikeExecutor.executeF32ToBF16(SoftmaxLikeReduction.LOG_SOFTMAX, input, continuation, node, reduction.getDimension(), context);
            return;
        }
        SoftmaxLikeExecutor.executeBF16(SoftmaxLikeReduction.LOG_SOFTMAX, input, node, reduction.getDimension(), context);
    }

    private static logSoftmax require(Operation op) {
        if (!(op instanceof logSoftmax reduction)) {
            throw new IllegalArgumentException("CpuLogSoftmaxKernel requires logSoftmax operation");
        }
        return reduction;
    }
}
