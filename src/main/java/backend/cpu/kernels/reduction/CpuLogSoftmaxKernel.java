package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.reduction.logSoftmax;
import tensor.Tensor;

public final class CpuLogSoftmaxKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        logSoftmax reduction = require(call.operation());
        Tensor input = CpuSoftmaxKernel.requireSingleInput(call.inputTensors());
        Tensor node = call.outputTensor();
        CpuKernelContext context = call.context();
        switch (node.getDataType()) {
            case FLOAT64 ->
                    SoftmaxLikeExecutor.executeF64(SoftmaxLikeReduction.LOG_SOFTMAX, input, node, reduction.getDimension(), context);
            case FLOAT32 ->
                    SoftmaxLikeExecutor.executeF32(SoftmaxLikeReduction.LOG_SOFTMAX, input, node, reduction.getDimension(), context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, input.getFlatDataSize());
                if (context.publishFloatContinuation() && context.cpuWorkspace() != null && continuation != null) {
                    float[] out = context.cpuWorkspace().requireFloatWorkspace();
                    SoftmaxLikeExecutor.executeF32ToFloat(SoftmaxLikeReduction.LOG_SOFTMAX, input, continuation, out, reduction.getDimension(), context);
                    context.cpuWorkspace().publishFloatContinuation(input.getFlatDataSize());
                    return CpuKernelResult.completed();
                }
                if (continuation != null) {
                    SoftmaxLikeExecutor.executeF32ToBF16(SoftmaxLikeReduction.LOG_SOFTMAX, input, continuation, node, reduction.getDimension(), context);
                } else {
                    SoftmaxLikeExecutor.executeBF16(SoftmaxLikeReduction.LOG_SOFTMAX, input, node, reduction.getDimension(), context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static logSoftmax require(Operation op) {
        if (!(op instanceof logSoftmax reduction)) {
            throw new IllegalArgumentException("CpuLogSoftmaxKernel requires logSoftmax operation");
        }
        return reduction;
    }
}
