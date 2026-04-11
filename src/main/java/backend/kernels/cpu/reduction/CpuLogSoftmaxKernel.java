package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.reduction.LogSoftmaxExecutor;
import operations.Operation;
import operations.logSoftmax;
import tensor.Tensor;

import java.util.List;

public final class CpuLogSoftmaxKernel implements CpuKernel {
    private static final LogSoftmaxExecutor EXECUTOR = new LogSoftmaxExecutor();

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof logSoftmax reduction)) {
            throw new IllegalArgumentException("CpuLogSoftmaxKernel requires logSoftmax operation");
        }
        EXECUTOR.execute(reduction, requireSingleInput(inputs), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof logSoftmax reduction)) {
            throw new IllegalArgumentException("CpuLogSoftmaxKernel requires logSoftmax operation");
        }
        EXECUTOR.executeF32(reduction, requireSingleInput(inputs), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof logSoftmax reduction)) {
            throw new IllegalArgumentException("CpuLogSoftmaxKernel requires logSoftmax operation");
        }
        Tensor input = requireSingleInput(inputs);
        float[] continuation = context.inputFloatContinuation(0, input.getFlatDataSize());
        if (context.publishFloatContinuation() && context.cpuWorkspace() != null && continuation != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            backend.kernels.cpu.reduction.LogSoftmaxLoops.executeF32ToFloat(input, continuation, out, reduction.getDimension(), context);
            context.cpuWorkspace().publishFloatContinuation(input.getFlatDataSize());
            return;
        }
        if (continuation != null) {
            backend.kernels.cpu.reduction.LogSoftmaxLoops.executeF32ToBF16(input, continuation, node, reduction.getDimension(), context);
            return;
        }
        EXECUTOR.executeBF16(reduction, input, node, context);
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("LogSoftmax expects exactly one input tensor");
        }
        return inputs.getFirst();
    }
}
