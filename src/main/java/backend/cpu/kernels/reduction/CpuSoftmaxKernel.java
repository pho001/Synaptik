package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.reduction.softmax;
import tensor.Tensor;

import java.util.List;

public final class CpuSoftmaxKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        softmax reduction = require(op);
        Tensor input = requireSingleInput(inputs);
        SoftmaxLikeExecutor.executeF64(SoftmaxLikeReduction.SOFTMAX, input, node, reduction.getDimension(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        softmax reduction = require(op);
        Tensor input = requireSingleInput(inputs);
        SoftmaxLikeExecutor.executeF32(SoftmaxLikeReduction.SOFTMAX, input, node, reduction.getDimension(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        softmax reduction = require(op);
        Tensor input = requireSingleInput(inputs);
        float[] continuation = context.inputFloatContinuation(0, input.getFlatDataSize());
        if (context.publishFloatContinuation() && context.cpuWorkspace() != null && continuation != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            SoftmaxLikeExecutor.executeF32ToFloat(SoftmaxLikeReduction.SOFTMAX, input, continuation, out, reduction.getDimension(), context);
            context.cpuWorkspace().publishFloatContinuation(input.getFlatDataSize());
            return;
        }
        if (continuation != null) {
            SoftmaxLikeExecutor.executeF32ToBF16(SoftmaxLikeReduction.SOFTMAX, input, continuation, node, reduction.getDimension(), context);
            return;
        }
        SoftmaxLikeExecutor.executeBF16(SoftmaxLikeReduction.SOFTMAX, input, node, reduction.getDimension(), context);
    }

    private static softmax require(Operation op) {
        if (!(op instanceof softmax reduction)) {
            throw new IllegalArgumentException("CpuSoftmaxKernel requires softmax operation");
        }
        return reduction;
    }

    static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Softmax expects exactly one input tensor");
        }
        return inputs.getFirst();
    }
}
