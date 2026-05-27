package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.loss.crossEntropyLossIndices;
import tensor.Tensor;

import java.util.List;

public final class CpuCrossEntropyLossIndicesKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        crossEntropyLossIndices loss = require(call.operation());
        Tensor[] pair = requirePair(call.inputTensors());
        Tensor node = call.outputTensor();
        CpuKernelContext context = call.context();
        switch (node.getDataType()) {
            case FLOAT64 -> CrossEntropyLossIndicesExecutor.executeF64(loss, pair[0], pair[1], node, context);
            case FLOAT32 -> CrossEntropyLossIndicesExecutor.executeF32(loss, pair[0], pair[1], node, context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, pair[0].getFlatDataSize());
                if (continuation != null) {
                    CrossEntropyLossIndicesExecutor.executeF32ToBF16(loss, pair[0], continuation, pair[1], node, context);
                } else {
                    CrossEntropyLossIndicesExecutor.executeBF16(loss, pair[0], pair[1], node, context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static crossEntropyLossIndices require(Operation op) {
        if (!(op instanceof crossEntropyLossIndices loss)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossIndicesKernel requires crossEntropyLossIndices operation");
        }
        return loss;
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Cross entropy loss from indices expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
