package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.reduction.NllLossExecutor;
import operations.Operation;
import operations.nllLoss;
import tensor.Tensor;

import java.util.List;

public final class CpuNllLossKernel implements CpuKernel {
    private static final NllLossExecutor EXECUTOR = new NllLossExecutor();

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof nllLoss loss)) {
            throw new IllegalArgumentException("CpuNllLossKernel requires nllLoss operation");
        }
        Tensor[] pair = requirePair(inputs);
        EXECUTOR.execute(loss, pair[0], pair[1], node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof nllLoss loss)) {
            throw new IllegalArgumentException("CpuNllLossKernel requires nllLoss operation");
        }
        Tensor[] pair = requirePair(inputs);
        EXECUTOR.executeF32(loss, pair[0], pair[1], node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof nllLoss loss)) {
            throw new IllegalArgumentException("CpuNllLossKernel requires nllLoss operation");
        }
        Tensor[] pair = requirePair(inputs);
        float[] continuation = context.inputFloatContinuation(0, pair[0].getFlatDataSize());
        if (continuation != null && pair[1].getDataType() == tensor.DataType.BFLOAT16) {
            backend.kernels.cpu.reduction.NllLossLoops.executeF32ToBF16(lossInput(pair[0]), continuation, pair[1], node, loss.getClassDimension(), context);
            return;
        }
        EXECUTOR.executeBF16(loss, pair[0], pair[1], node, context);
    }

    private static Tensor lossInput(Tensor tensor) {
        return tensor;
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("NLL loss expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
