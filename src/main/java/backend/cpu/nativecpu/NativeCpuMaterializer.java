package backend.cpu.nativecpu;

import tensor.BFloat16Storage;
import tensor.DataType;
import tensor.Float32Storage;
import tensor.Float64Storage;
import tensor.NativeBFloat16Storage;
import tensor.NativeFloat32Storage;
import tensor.NativeFloat64Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Copies tensor data between Java array storage and native CPU storage.
 */
public final class NativeCpuMaterializer {
    private NativeCpuMaterializer() {
    }

    public static void arrayToNative(Tensor source, NativeTensorStorage target) {
        Objects.requireNonNull(source, "source cannot be null");
        Objects.requireNonNull(target, "target cannot be null");
        validateDenseContiguous(source, "array -> native");
        validateShapeAndType(source, target);
        switch (source.getDataType()) {
            case FLOAT32 -> copyF32ArrayToNative(source, target);
            case FLOAT64 -> copyF64ArrayToNative(source, target);
            case BFLOAT16 -> copyBF16ArrayToNative(source, target);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("array -> native MVP supports only FLOAT32, FLOAT64, and BFLOAT16. dtype=" + source.getDataType());
        }
        target.markModified();
    }

    public static void nativeToArray(NativeTensorStorage source, Tensor target) {
        Objects.requireNonNull(source, "source cannot be null");
        Objects.requireNonNull(target, "target cannot be null");
        validateDenseContiguous(target, "native -> array");
        validateShapeAndType(target, source);
        switch (target.getDataType()) {
            case FLOAT32 -> copyF32NativeToArray(source, target);
            case FLOAT64 -> copyF64NativeToArray(source, target);
            case BFLOAT16 -> copyBF16NativeToArray(source, target);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("native -> array MVP supports only FLOAT32, FLOAT64, and BFLOAT16. dtype=" + target.getDataType());
        }
        target.markStorageModified();
    }

    private static void copyF32ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeFloat32Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        float[] data = source.getFloat32Data();
        if (data == null) {
            throw new IllegalStateException("FLOAT32 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_FLOAT, 0L, source.getFlatDataSize());
    }

    private static void copyF64ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeFloat64Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        double[] data = source.getFloat64Data();
        if (data == null) {
            throw new IllegalStateException("FLOAT64 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_DOUBLE, 0L, source.getFlatDataSize());
    }

    private static void copyBF16ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeBFloat16Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        short[] data = source.getBFloat16Data();
        if (data == null) {
            throw new IllegalStateException("BFLOAT16 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_SHORT, 0L, source.getFlatDataSize());
    }

    private static void copyF32NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeFloat32Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        float[] data = target.getFloat32Data();
        if (data == null) {
            throw new IllegalStateException("FLOAT32 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_FLOAT, 0L, data, 0, target.getFlatDataSize());
    }

    private static void copyF64NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeFloat64Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        double[] data = target.getFloat64Data();
        if (data == null) {
            throw new IllegalStateException("FLOAT64 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_DOUBLE, 0L, data, 0, target.getFlatDataSize());
    }

    private static void copyBF16NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeBFloat16Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        short[] data = target.getBFloat16Data();
        if (data == null) {
            throw new IllegalStateException("BFLOAT16 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_SHORT, 0L, data, 0, target.getFlatDataSize());
    }

    private static void validateShapeAndType(Tensor tensor, NativeTensorStorage storage) {
        if (tensor.getDataType() != storage.getType()) {
            throw typeMismatch(tensor.getDataType(), storage.getType());
        }
        if (tensor.getFlatDataSize() != storage.getSize()) {
            throw new IllegalArgumentException("Tensor/native storage size mismatch. tensorElements="
                    + tensor.getFlatDataSize() + ", nativeElements=" + storage.getSize());
        }
    }

    private static void validateDenseContiguous(Tensor tensor, String direction) {
        if (!tensor.isContiguous() || tensor.hasStorageOffset()) {
            throw new UnsupportedOperationException(direction
                    + " native CPU materialization MVP supports only dense contiguous tensors without storageOffset. shape="
                    + Arrays.toString(tensor.getShapeUnsafe())
                    + ", strides=" + Arrays.toString(tensor.getStridesUnsafe())
                    + ", storageOffset=" + tensor.getStorageOffsetUnsafe());
        }
    }

    private static IllegalArgumentException typeMismatch(DataType tensorType, DataType storageType) {
        return new IllegalArgumentException("Tensor/native storage dtype mismatch. tensorType="
                + tensorType + ", nativeType=" + storageType);
    }
}
