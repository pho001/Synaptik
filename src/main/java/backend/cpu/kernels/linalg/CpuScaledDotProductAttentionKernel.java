package backend.cpu.kernels.linalg;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import tensor.Tensor;

import java.util.List;

public final class CpuScaledDotProductAttentionKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scaledDotProductAttention attention = require(op);
        ScaledDotProductAttentionExecutor.executeF64(attention, requireInputs(inputs, attention), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scaledDotProductAttention attention = require(op);
        ScaledDotProductAttentionExecutor.executeF32(attention, requireInputs(inputs, attention), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scaledDotProductAttention attention = require(op);
        ScaledDotProductAttentionExecutor.executeBF16(attention, requireInputs(inputs, attention), node, context);
    }

    private static scaledDotProductAttention require(Operation op) {
        if (!(op instanceof scaledDotProductAttention attention)) {
            throw new IllegalArgumentException("CpuScaledDotProductAttentionKernel requires scaledDotProductAttention operation");
        }
        return attention;
    }

    private static Tensor[] requireInputs(List<Tensor> inputs, scaledDotProductAttention attention) {
        int expected = attention.hasMask() ? 4 : 3;
        if (inputs == null || inputs.size() != expected) {
            throw new IllegalArgumentException("scaledDotProductAttention expects exactly " + expected + " inputs");
        }
        Tensor[] out = new Tensor[expected];
        for (int i = 0; i < expected; i++) {
            out[i] = inputs.get(i);
        }
        return out;
    }
}
