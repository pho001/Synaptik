package backend.cpu.kernels.linalg;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public final class CpuScaledDotProductAttentionKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        scaledDotProductAttention attention = require(call.operation());
        Tensor node = call.outputTensor();
        Tensor[] inputTensors = requireInputTensors(call.inputTensors(), attention);
        CpuStorageView[] inputViews = requireInputViews(call.inputs(), attention);
        CpuStorageView output = requireOutput(call.output(), node);
        switch (output.dtype()) {
            case FLOAT64 -> ScaledDotProductAttentionExecutor.executeF64(attention, inputTensors, inputViews, node, output, call.context());
            case FLOAT32 -> ScaledDotProductAttentionExecutor.executeF32(attention, inputTensors, inputViews, node, output, call.context());
            case BFLOAT16 -> ScaledDotProductAttentionExecutor.executeBF16(attention, inputTensors, inputViews, node, output, call.context());
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static scaledDotProductAttention require(Operation op) {
        if (!(op instanceof scaledDotProductAttention attention)) {
            throw new IllegalArgumentException("CpuScaledDotProductAttentionKernel requires scaledDotProductAttention operation");
        }
        return attention;
    }

    private static Tensor[] requireInputTensors(List<Tensor> inputs, scaledDotProductAttention attention) {
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

    private static CpuStorageView[] requireInputViews(List<CpuStorageView> inputs, scaledDotProductAttention attention) {
        int expected = attention.hasMask() ? 4 : 3;
        if (inputs == null || inputs.size() != expected) {
            throw new IllegalArgumentException("scaledDotProductAttention expects exactly " + expected + " input storage views");
        }
        CpuStorageView[] out = new CpuStorageView[expected];
        for (int i = 0; i < expected; i++) {
            out[i] = inputs.get(i);
            if (out[i] == null) {
                throw new IllegalArgumentException("scaledDotProductAttention input storage view " + i + " cannot be null");
            }
        }
        return out;
    }

    private static CpuStorageView requireOutput(CpuStorageView output, Tensor node) {
        if (output == null) {
            throw new IllegalArgumentException("scaledDotProductAttention requires an output storage view");
        }
        if (output.dtype() != node.getDataType()) {
            throw new IllegalArgumentException("scaledDotProductAttention output dtype mismatch: view="
                    + output.dtype() + ", tensor=" + node.getDataType());
        }
        if (!Arrays.equals(output.shape(), node.getShapeUnsafe())) {
            throw new IllegalArgumentException("scaledDotProductAttention output shape mismatch");
        }
        return output;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuScaledDotProductAttentionKernel does not support " + dtype);
    }
}
