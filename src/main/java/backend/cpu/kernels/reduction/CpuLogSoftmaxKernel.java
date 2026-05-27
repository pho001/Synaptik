package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.reduction.logSoftmax;

public final class CpuLogSoftmaxKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        logSoftmax reduction = require(call.operation());
        CpuSoftmaxKernel.requireSingleInput(call.inputTensors());
        CpuStorageView inputView = CpuSoftmaxKernel.requireSingleInputView(call, "LogSoftmax");
        CpuStorageView outputView = CpuSoftmaxKernel.requireOutputView(call, "LogSoftmax");
        CpuKernelContext context = call.context();
        switch (outputView.dtype()) {
            case FLOAT64 ->
                    SoftmaxLikeExecutor.executeF64(SoftmaxLikeReduction.LOG_SOFTMAX, inputView, outputView, reduction.getDimension(), context);
            case FLOAT32 ->
                    SoftmaxLikeExecutor.executeF32(SoftmaxLikeReduction.LOG_SOFTMAX, inputView, outputView, reduction.getDimension(), context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, inputView.logicalSize());
                if (context.publishFloatContinuation() && context.cpuWorkspace() != null && continuation != null) {
                    float[] out = context.cpuWorkspace().requireFloatWorkspace();
                    SoftmaxLikeExecutor.executeF32ToFloat(SoftmaxLikeReduction.LOG_SOFTMAX, inputView, continuation, out, reduction.getDimension(), context);
                    context.cpuWorkspace().publishFloatContinuation(inputView.logicalSize());
                    return CpuKernelResult.completed();
                }
                if (continuation != null) {
                    SoftmaxLikeExecutor.executeF32ToBF16(SoftmaxLikeReduction.LOG_SOFTMAX, inputView, continuation, outputView, reduction.getDimension(), context);
                } else {
                    SoftmaxLikeExecutor.executeBF16(SoftmaxLikeReduction.LOG_SOFTMAX, inputView, outputView, reduction.getDimension(), context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + outputView.dtype());
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
