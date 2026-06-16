package backend.cpu.nativecpu;

import tensor.TensorInternalAccess;

import tensor.storage.BFloat16Storage;
import tensor.DataType;
import tensor.storage.Float32Storage;
import tensor.storage.Float64Storage;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeBoolStorage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeInt32Storage;
import tensor.storage.NativeInt64Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

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
            case BOOL -> copyBoolArrayToNative(source, target);
            case INT32 -> copyI32ArrayToNative(source, target);
            case INT64 -> copyI64ArrayToNative(source, target);
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
            case BOOL -> copyBoolNativeToArray(source, target);
            case INT32 -> copyI32NativeToArray(source, target);
            case INT64 -> copyI64NativeToArray(source, target);
        }
        TensorInternalAccess.markStorageModified(target);
    }

    private static void copyF32ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeFloat32Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        float[] data = TensorInternalAccess.float32Data(source);
        if (data == null) {
            throw new IllegalStateException("FLOAT32 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_FLOAT, 0L, source.getFlatDataSize());
    }

    private static void copyF64ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeFloat64Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        double[] data = TensorInternalAccess.float64Data(source);
        if (data == null) {
            throw new IllegalStateException("FLOAT64 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_DOUBLE, 0L, source.getFlatDataSize());
    }

    private static void copyBF16ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeBFloat16Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        short[] data = TensorInternalAccess.bfloat16Data(source);
        if (data == null) {
            throw new IllegalStateException("BFLOAT16 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_SHORT, 0L, source.getFlatDataSize());
    }

    private static void copyBoolArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeBoolStorage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        byte[] data = TensorInternalAccess.boolData(source);
        if (data == null) {
            throw new IllegalStateException("BOOL source does not expose CPU array storage.");
        }
        MemorySegment segment = target.segment();
        for (int i = 0; i < source.getFlatDataSize(); i++) {
            segment.set(JAVA_BYTE, i, data[i] == 0 ? (byte) 0 : (byte) 1);
        }
    }

    private static void copyI32ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeInt32Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        int[] data = TensorInternalAccess.int32Data(source);
        if (data == null) {
            throw new IllegalStateException("INT32 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_INT, 0L, source.getFlatDataSize());
    }

    private static void copyI64ArrayToNative(Tensor source, NativeTensorStorage target) {
        if (!(target instanceof NativeInt64Storage)) {
            throw typeMismatch(source.getDataType(), target.getType());
        }
        long[] data = TensorInternalAccess.int64Data(source);
        if (data == null) {
            throw new IllegalStateException("INT64 source does not expose CPU array storage.");
        }
        MemorySegment.copy(data, 0, target.segment(), JAVA_LONG, 0L, source.getFlatDataSize());
    }

    private static void copyF32NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeFloat32Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        float[] data = TensorInternalAccess.float32Data(target);
        if (data == null) {
            throw new IllegalStateException("FLOAT32 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_FLOAT, 0L, data, 0, target.getFlatDataSize());
    }

    private static void copyF64NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeFloat64Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        double[] data = TensorInternalAccess.float64Data(target);
        if (data == null) {
            throw new IllegalStateException("FLOAT64 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_DOUBLE, 0L, data, 0, target.getFlatDataSize());
    }

    private static void copyBF16NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeBFloat16Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        short[] data = TensorInternalAccess.bfloat16Data(target);
        if (data == null) {
            throw new IllegalStateException("BFLOAT16 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_SHORT, 0L, data, 0, target.getFlatDataSize());
    }

    private static void copyBoolNativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeBoolStorage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        byte[] data = TensorInternalAccess.boolData(target);
        if (data == null) {
            throw new IllegalStateException("BOOL target does not expose CPU array storage.");
        }
        MemorySegment segment = source.segment();
        for (int i = 0; i < target.getFlatDataSize(); i++) {
            data[i] = segment.get(JAVA_BYTE, i) == 0 ? (byte) 0 : (byte) 1;
        }
    }

    private static void copyI32NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeInt32Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        int[] data = TensorInternalAccess.int32Data(target);
        if (data == null) {
            throw new IllegalStateException("INT32 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_INT, 0L, data, 0, target.getFlatDataSize());
    }

    private static void copyI64NativeToArray(NativeTensorStorage source, Tensor target) {
        if (!(source instanceof NativeInt64Storage)) {
            throw typeMismatch(target.getDataType(), source.getType());
        }
        long[] data = TensorInternalAccess.int64Data(target);
        if (data == null) {
            throw new IllegalStateException("INT64 target does not expose CPU array storage.");
        }
        MemorySegment.copy(source.segment(), JAVA_LONG, 0L, data, 0, target.getFlatDataSize());
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
