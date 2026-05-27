package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.loss.nllLoss;
import tensor.Tensor;

import java.util.List;

public final class CpuNllLossKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        nllLoss loss = require(call.operation());
        Tensor[] pair = requirePair(call.inputTensors(), "NLL loss");
        Tensor node = call.outputTensor();
        CpuKernelContext context = call.context();
        switch (node.getDataType()) {
            case FLOAT64 -> LossReductionExecutor.executeF64(
                    LossReduction.NLL, pair[0], pair[1], node, loss.getClassDimension(), context);
            case FLOAT32 -> LossReductionExecutor.executeF32(
                    LossReduction.NLL, pair[0], pair[1], node, loss.getClassDimension(), context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, pair[0].getFlatDataSize());
                if (continuation != null) {
                    LossReductionExecutor.executeF32ToBF16(LossReduction.NLL, pair[0], continuation, pair[1], node, loss.getClassDimension(), context);
                } else {
                    LossReductionExecutor.executeBF16(LossReduction.NLL, pair[0], pair[1], node, loss.getClassDimension(), context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static nllLoss require(Operation op) {
        if (!(op instanceof nllLoss loss)) {
            throw new IllegalArgumentException("CpuNllLossKernel requires nllLoss operation");
        }
        return loss;
    }

    static Tensor[] requirePair(List<Tensor> inputs, String label) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException(label + " expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
