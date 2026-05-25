package backend.cpu.kernels.reduction;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.loss.nllLoss;
import tensor.Tensor;

import java.util.List;

public final class CpuNllLossKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        nllLoss loss = require(op);
        Tensor[] pair = requirePair(inputs, "NLL loss");
        LossReductionExecutor.executeF64(LossReduction.NLL, pair[0], pair[1], node, loss.getClassDimension(), context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        nllLoss loss = require(op);
        Tensor[] pair = requirePair(inputs, "NLL loss");
        LossReductionExecutor.executeF32(LossReduction.NLL, pair[0], pair[1], node, loss.getClassDimension(), context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        nllLoss loss = require(op);
        Tensor[] pair = requirePair(inputs, "NLL loss");
        float[] continuation = context.inputFloatContinuation(0, pair[0].getFlatDataSize());
        if (continuation != null) {
            LossReductionExecutor.executeF32ToBF16(LossReduction.NLL, pair[0], continuation, pair[1], node, loss.getClassDimension(), context);
            return;
        }
        LossReductionExecutor.executeBF16(LossReduction.NLL, pair[0], pair[1], node, loss.getClassDimension(), context);
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
