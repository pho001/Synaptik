package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import operations.crossEntropyLossIndices;
import tensor.Tensor;

import java.util.List;

public final class CpuCrossEntropyLossIndicesKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLossIndices loss = require(op);
        Tensor[] pair = requirePair(inputs);
        CrossEntropyLossIndicesExecutor.executeF64(loss, pair[0], pair[1], node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLossIndices loss = require(op);
        Tensor[] pair = requirePair(inputs);
        CrossEntropyLossIndicesExecutor.executeF32(loss, pair[0], pair[1], node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        crossEntropyLossIndices loss = require(op);
        Tensor[] pair = requirePair(inputs);
        float[] continuation = context.inputFloatContinuation(0, pair[0].getFlatDataSize());
        if (continuation != null) {
            CrossEntropyLossIndicesExecutor.executeF32ToBF16(loss, pair[0], continuation, pair[1], node, context);
            return;
        }
        CrossEntropyLossIndicesExecutor.executeBF16(loss, pair[0], pair[1], node, context);
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
