package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.loss.crossEntropyLoss;

public final class CpuCrossEntropyLossKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        crossEntropyLoss loss = require(call.operation());
        CpuNllLossKernel.requirePair(call.inputTensors(), "Cross entropy loss");
        CpuStorageView[] pair = CpuNllLossKernel.requireViewPair(call.inputs(), "Cross entropy loss");
        CpuStorageView output = CpuSoftmaxKernel.requireOutputView(call, "Cross entropy loss");
        CpuKernelContext context = call.context();
        switch (output.dtype()) {
            case FLOAT64 -> LossReductionExecutor.executeF64(
                    LossReduction.CROSS_ENTROPY, pair[0], pair[1], output, loss.getClassDimension(), context);
            case FLOAT32 -> LossReductionExecutor.executeF32(
                    LossReduction.CROSS_ENTROPY, pair[0], pair[1], output, loss.getClassDimension(), context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, pair[0].logicalSize());
                if (continuation != null) {
                    LossReductionExecutor.executeF32ToBF16(LossReduction.CROSS_ENTROPY, pair[0], continuation, pair[1], output, loss.getClassDimension(), context);
                } else {
                    LossReductionExecutor.executeBF16(LossReduction.CROSS_ENTROPY, pair[0], pair[1], output, loss.getClassDimension(), context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static crossEntropyLoss require(Operation op) {
        if (!(op instanceof crossEntropyLoss loss)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossKernel requires crossEntropyLoss operation");
        }
        return loss;
    }
}
