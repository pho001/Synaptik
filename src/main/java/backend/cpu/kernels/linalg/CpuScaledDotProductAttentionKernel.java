package backend.cpu.kernels.linalg;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class CpuScaledDotProductAttentionKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        scaledDotProductAttention attention = require(call.operation());
        Tensor node = call.outputTensor();
        Tensor[] inputs = requireInputs(call.inputTensors(), attention);
        switch (node.getDataType()) {
            case FLOAT64 -> ScaledDotProductAttentionExecutor.executeF64(attention, inputs, node, call.context());
            case FLOAT32 -> ScaledDotProductAttentionExecutor.executeF32(attention, inputs, node, call.context());
            case BFLOAT16 -> ScaledDotProductAttentionExecutor.executeBF16(attention, inputs, node, call.context());
            case INT32, INT64, BOOL -> unsupported(node.getDataType());
        }
        return CpuKernelResult.completed();
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

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuScaledDotProductAttentionKernel does not support " + dtype);
    }
}
