package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.loss.crossEntropyLossIndices;

public final class CpuCrossEntropyLossIndicesKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        crossEntropyLossIndices loss = require(call.operation());
        CpuNllLossKernel.requirePair(call.inputTensors(), "Cross entropy loss from indices");
        CpuStorageView[] pair = CpuNllLossKernel.requireViewPair(call.inputs(), "Cross entropy loss from indices");
        CpuStorageView output = CpuSoftmaxKernel.requireOutputView(call, "Cross entropy loss from indices");
        CpuKernelContext context = call.context();
        switch (output.dtype()) {
            case FLOAT64 -> CrossEntropyLossIndicesExecutor.executeF64(loss, pair[0], pair[1], output, context);
            case FLOAT32 -> CrossEntropyLossIndicesExecutor.executeF32(loss, pair[0], pair[1], output, context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, pair[0].logicalSize());
                if (continuation != null) {
                    CrossEntropyLossIndicesExecutor.executeF32ToBF16(loss, pair[0], continuation, pair[1], output, context);
                } else {
                    CrossEntropyLossIndicesExecutor.executeBF16(loss, pair[0], pair[1], output, context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static crossEntropyLossIndices require(Operation op) {
        if (!(op instanceof crossEntropyLossIndices loss)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossIndicesKernel requires crossEntropyLossIndices operation");
        }
        return loss;
    }
}
