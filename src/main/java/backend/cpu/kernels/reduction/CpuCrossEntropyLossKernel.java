package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.loss.crossEntropyLoss;
import tensor.Tensor;

import java.util.List;

public final class CpuCrossEntropyLossKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLoss loss = require(op);
        Tensor[] pair = CpuNllLossKernel.requirePair(inputs, "Cross entropy loss");
        LossReductionExecutor.executeF64(LossReduction.CROSS_ENTROPY, pair[0], pair[1], node, loss.getClassDimension(), context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLoss loss = require(op);
        Tensor[] pair = CpuNllLossKernel.requirePair(inputs, "Cross entropy loss");
        LossReductionExecutor.executeF32(LossReduction.CROSS_ENTROPY, pair[0], pair[1], node, loss.getClassDimension(), context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLoss loss = require(op);
        Tensor[] pair = CpuNllLossKernel.requirePair(inputs, "Cross entropy loss");
        float[] continuation = context.inputFloatContinuation(0, pair[0].getFlatDataSize());
        if (continuation != null) {
            LossReductionExecutor.executeF32ToBF16(LossReduction.CROSS_ENTROPY, pair[0], continuation, pair[1], node, loss.getClassDimension(), context);
            return;
        }
        LossReductionExecutor.executeBF16(LossReduction.CROSS_ENTROPY, pair[0], pair[1], node, loss.getClassDimension(), context);
    }

    private static crossEntropyLoss require(Operation op) {
        if (!(op instanceof crossEntropyLoss loss)) {
            throw new IllegalArgumentException("CpuCrossEntropyLossKernel requires crossEntropyLoss operation");
        }
        return loss;
    }
}
