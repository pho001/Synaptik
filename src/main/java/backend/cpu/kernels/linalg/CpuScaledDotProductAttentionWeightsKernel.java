package backend.cpu.kernels.linalg;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageResolver;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.linalg.scaledDotProductAttentionWeights;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuScaledDotProductAttentionWeightsKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        Tensor node = call.outputTensor();
        CpuStorageView output = requireOutputView(call, node);
        Tensor cached = requireCachedWeights(call.operation(), call.inputTensors(), node, call.context());
        CpuStorageView source = new CpuStorageResolver().bindArrayOnly(cached);
        switch (output.dtype()) {
            case FLOAT64 -> publishF64(source, output);
            case FLOAT32 -> publishF32(source, output);
            case BFLOAT16 -> publishBF16(source, output);
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static void publishF64(CpuStorageView source, CpuStorageView output) {
        if (source.dtype() != DataType.FLOAT64) {
            throw new IllegalStateException("FLOAT64 attention weights cache required, actual=" + source.dtype());
        }
        int size = output.logicalSize();
        if (isDenseZero(source) && isDenseZero(output)) {
            if (output.isArray()) {
                System.arraycopy(source.requireF64Array(), source.storageOffset(), output.requireF64Array(), output.storageOffset(), size);
            } else {
                MemorySegment.copy(source.requireF64Array(), source.storageOffset(), output.requireSegment(),
                        JAVA_DOUBLE, (long) output.storageOffset() * Double.BYTES, size);
            }
            return;
        }
        double[] src = source.requireF64Array();
        int[] srcShape = source.shape();
        int[] srcStrides = source.strides();
        int[] outShape = output.shape();
        int[] outStrides = output.strides();
        for (int i = 0; i < size; i++) {
            writeF64(output, physicalOffset(outShape, outStrides, output.storageOffset(), i),
                    src[physicalOffset(srcShape, srcStrides, source.storageOffset(), i)]);
        }
    }

    private static void publishF32(CpuStorageView source, CpuStorageView output) {
        if (source.dtype() != DataType.FLOAT32) {
            throw new IllegalStateException("FLOAT32 attention weights cache required, actual=" + source.dtype());
        }
        int size = output.logicalSize();
        if (isDenseZero(source) && isDenseZero(output)) {
            if (output.isArray()) {
                System.arraycopy(source.requireF32Array(), source.storageOffset(), output.requireF32Array(), output.storageOffset(), size);
            } else {
                MemorySegment.copy(source.requireF32Array(), source.storageOffset(), output.requireSegment(),
                        JAVA_FLOAT, (long) output.storageOffset() * Float.BYTES, size);
            }
            return;
        }
        float[] src = source.requireF32Array();
        int[] srcShape = source.shape();
        int[] srcStrides = source.strides();
        int[] outShape = output.shape();
        int[] outStrides = output.strides();
        for (int i = 0; i < size; i++) {
            writeF32(output, physicalOffset(outShape, outStrides, output.storageOffset(), i),
                    src[physicalOffset(srcShape, srcStrides, source.storageOffset(), i)]);
        }
    }

    private static void publishBF16(CpuStorageView source, CpuStorageView output) {
        if (source.dtype() == DataType.BFLOAT16) {
            publishBF16Bits(source, output);
            return;
        }
        if (source.dtype() == DataType.FLOAT32) {
            publishF32ToBF16(source, output);
            return;
        }
        throw new IllegalStateException("Unsupported cached weights dtype for BF16 export: " + source.dtype());
    }

    private static void publishBF16Bits(CpuStorageView source, CpuStorageView output) {
        int size = output.logicalSize();
        if (isDenseZero(source) && isDenseZero(output)) {
            if (output.isArray()) {
                System.arraycopy(source.requireBF16Array(), source.storageOffset(), output.requireBF16Array(), output.storageOffset(), size);
            } else {
                MemorySegment.copy(source.requireBF16Array(), source.storageOffset(), output.requireSegment(),
                        JAVA_SHORT, (long) output.storageOffset() * Short.BYTES, size);
            }
            return;
        }
        short[] src = source.requireBF16Array();
        int[] srcShape = source.shape();
        int[] srcStrides = source.strides();
        int[] outShape = output.shape();
        int[] outStrides = output.strides();
        for (int i = 0; i < size; i++) {
            writeBF16(output, physicalOffset(outShape, outStrides, output.storageOffset(), i),
                    src[physicalOffset(srcShape, srcStrides, source.storageOffset(), i)]);
        }
    }

    private static void publishF32ToBF16(CpuStorageView source, CpuStorageView output) {
        int size = output.logicalSize();
        float[] src = source.requireF32Array();
        int[] srcShape = source.shape();
        int[] srcStrides = source.strides();
        int[] outShape = output.shape();
        int[] outStrides = output.strides();
        for (int i = 0; i < size; i++) {
            short bits = TensorDTypeOps.toBFloat16Bits(src[physicalOffset(srcShape, srcStrides, source.storageOffset(), i)]);
            writeBF16(output, physicalOffset(outShape, outStrides, output.storageOffset(), i), bits);
        }
    }

    private static Tensor requireCachedWeights(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof scaledDotProductAttentionWeights)) {
            throw new IllegalArgumentException("CpuScaledDotProductAttentionWeightsKernel requires scaledDotProductAttentionWeights operation");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("scaledDotProductAttentionWeights expects exactly one input");
        }
        Tensor attentionOut = inputs.getFirst();
        ScaledDotProductAttentionRuntimeCache runtimeCache =
                context == null ? null : context.runtimeStateFor(attentionOut, ScaledDotProductAttentionRuntimeCache.class);
        if (runtimeCache == null) {
            throw new IllegalStateException("scaledDotProductAttentionWeights requires cached forward weights on the attention output tensor");
        }
        Tensor weights = runtimeCache.weights();
        if (weights.getDataType() != node.getDataType()
                && !(weights.getDataType() == DataType.FLOAT32 && node.getDataType() == DataType.BFLOAT16)) {
            throw new IllegalStateException("scaledDotProductAttentionWeights cache dtype mismatch: cache="
                    + weights.getDataType() + ", node=" + node.getDataType());
        }
        if (!Arrays.equals(weights.getShapeUnsafe(), node.getShapeUnsafe())) {
            throw new IllegalStateException("scaledDotProductAttentionWeights cache shape mismatch: cache="
                    + Arrays.toString(weights.getShapeUnsafe()) + ", node=" + Arrays.toString(node.getShapeUnsafe()));
        }
        return weights;
    }

    private static CpuStorageView requireOutputView(CpuKernelCall call, Tensor node) {
        CpuStorageView output = call.output();
        if (output == null) {
            throw new IllegalStateException("scaledDotProductAttentionWeights requires CpuStorageView output");
        }
        if (output.dtype() != node.getDataType()) {
            throw new IllegalStateException("scaledDotProductAttentionWeights output dtype mismatch: view="
                    + output.dtype() + ", node=" + node.getDataType());
        }
        if (!Arrays.equals(output.shape(), node.getShapeUnsafe())) {
            throw new IllegalStateException("scaledDotProductAttentionWeights output shape mismatch: view="
                    + Arrays.toString(output.shape()) + ", node=" + Arrays.toString(node.getShapeUnsafe()));
        }
        return output;
    }

    private static void writeF64(CpuStorageView output, int offset, double value) {
        if (output.isArray()) {
            output.requireF64Array()[offset] = value;
        } else {
            output.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static void writeF32(CpuStorageView output, int offset, float value) {
        if (output.isArray()) {
            output.requireF32Array()[offset] = value;
        } else {
            output.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static void writeBF16(CpuStorageView output, int offset, short value) {
        if (output.isArray()) {
            output.requireBF16Array()[offset] = value;
        } else {
            output.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, value);
        }
    }

    private static boolean isDenseZero(CpuStorageView view) {
        if (view.storageOffset() != 0) {
            return false;
        }
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[i]);
        }
        return true;
    }

    private static int physicalOffset(int[] shape, int[] strides, int storageOffset, int logicalIndex) {
        int offset = storageOffset;
        int remainder = logicalIndex;
        for (int i = shape.length - 1; i >= 0; i--) {
            int coordinate = remainder % shape[i];
            remainder /= shape[i];
            offset += coordinate * strides[i];
        }
        return offset;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuScaledDotProductAttentionWeightsKernel does not support " + dtype);
    }
}
