package tensor.storage;

import tensor.DataType;
import tensor.TensorMetadata;
import tensor.dtype.BFloat16Bits;

public final class TensorStorageSupport {
    private TensorStorageSupport() {
    }

    public static TensorStorage emptyStorage(TensorMetadata metadata) {
        return emptyStorage(metadata, metadata.getFlatSize());
    }

    public static TensorStorage emptyStorage(TensorMetadata metadata, int size) {
        DataType type = normalizeDataType(metadata);
        if (type == DataType.FLOAT64) {
            return new Float64Storage(size);
        }
        return switch (type) {
            case BOOL -> new BoolStorage(size);
            case BFLOAT16 -> new BFloat16Storage(size);
            case FLOAT32 -> new Float32Storage(size);
            case INT32 -> new Int32Storage(size);
            case INT64 -> new Int64Storage(size);
            case FLOAT64 -> throw new IllegalStateException("Unexpected dtype branch");
        };
    }

    public static TensorStorage fromDoubleArray(TensorMetadata metadata, double[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type == DataType.BOOL || type == DataType.INT32 || type == DataType.INT64) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32/INT64 storage conversion is not supported.");
        }
        int size = source.length;
        if (type == DataType.FLOAT64) {
            return new Float64Storage(source);
        }
        if (type == DataType.FLOAT32) {
            float[] converted = new float[size];
            for (int i = 0; i < size; i++) {
                converted[i] = (float) source[i];
            }
            return new Float32Storage(converted);
        }
        short[] converted = new short[size];
        for (int i = 0; i < size; i++) {
            converted[i] = BFloat16Bits.fromFloat((float) source[i]);
        }
        return new BFloat16Storage(converted);
    }

    public static TensorStorage fromFloatArray(TensorMetadata metadata, float[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type == DataType.BOOL || type == DataType.INT32 || type == DataType.INT64) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32/INT64 storage conversion is not supported.");
        }
        int size = source.length;
        return switch (type) {
            case FLOAT32 -> new Float32Storage(source);
            case FLOAT64 -> {
                double[] converted = new double[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = source[i];
                }
                yield new Float64Storage(converted);
            }
            case BFLOAT16 -> {
                short[] converted = new short[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = BFloat16Bits.fromFloat(source[i]);
                }
                yield new BFloat16Storage(converted);
            }
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32/INT64 storage conversion is not supported.");
        };
    }

    public static TensorStorage fromBFloat16Array(TensorMetadata metadata, short[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type == DataType.BOOL || type == DataType.INT32 || type == DataType.INT64) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32/INT64 storage conversion is not supported.");
        }
        int size = source.length;
        return switch (type) {
            case BFLOAT16 -> new BFloat16Storage(source);
            case FLOAT32 -> {
                float[] converted = new float[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = BFloat16Bits.toFloat(source[i]);
                }
                yield new Float32Storage(converted);
            }
            case FLOAT64 -> {
                double[] converted = new double[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = BFloat16Bits.toFloat(source[i]);
                }
                yield new Float64Storage(converted);
            }
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32/INT64 storage conversion is not supported.");
        };
    }

    public static TensorStorage fromBoolArray(TensorMetadata metadata, byte[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type != DataType.BOOL) {
            throw new UnsupportedOperationException("Implicit BOOL -> numeric storage conversion is not supported.");
        }
        byte[] normalized = new byte[source.length];
        for (int i = 0; i < source.length; i++) {
            normalized[i] = source[i] == 0 ? (byte) 0 : (byte) 1;
        }
        return new BoolStorage(normalized);
    }

    public static TensorStorage fromIntArray(TensorMetadata metadata, int[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type != DataType.INT32) {
            throw new UnsupportedOperationException("Implicit INT32 -> other dtype conversion is not supported.");
        }
        return new Int32Storage(source);
    }

    public static TensorStorage fromLongArray(TensorMetadata metadata, long[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type != DataType.INT64) {
            throw new UnsupportedOperationException("Implicit INT64 -> other dtype conversion is not supported.");
        }
        return new Int64Storage(source);
    }

    public static void validateInputLength(int actual, int expected, String sourceName) {
        if (actual != expected) {
            throw new IllegalArgumentException(sourceName + " length mismatch. expected=" + expected + ", actual=" + actual);
        }
    }

    public static double getByStorageOffset(TensorStorage storage, int storageSize, int offset) {
        if (storage == null) {
            throw new IllegalStateException("Tensor storage is not initialized.");
        }
        if (offset < 0 || offset >= storageSize) {
            throw new IndexOutOfBoundsException("Storage offset out of bounds.");
        }
        return switch (storage.getType()) {
            case FLOAT64 -> ((Float64Storage) storage).getDoubleArray()[offset];
            case FLOAT32 -> ((Float32Storage) storage).getFloatArray()[offset];
            case BFLOAT16 -> BFloat16Bits.toFloat(((BFloat16Storage) storage).getShortArray()[offset]);
            case INT32 -> ((Int32Storage) storage).getIntArray()[offset];
            case INT64 -> ((Int64Storage) storage).getLongArray()[offset];
            case BOOL -> ((BoolStorage) storage).getByteArray()[offset] == 0 ? 0.0d : 1.0d;
        };
    }

    public static void setByStorageOffset(TensorStorage storage, int storageSize, int offset, double value) {
        if (storage == null) {
            throw new IllegalStateException("Tensor storage is not initialized.");
        }
        if (offset < 0 || offset >= storageSize) {
            throw new IndexOutOfBoundsException("Storage offset out of bounds.");
        }
        switch (storage.getType()) {
            case FLOAT64 -> ((Float64Storage) storage).getDoubleArray()[offset] = value;
            case FLOAT32 -> ((Float32Storage) storage).getFloatArray()[offset] = (float) value;
            case BFLOAT16 -> ((BFloat16Storage) storage).getShortArray()[offset] = BFloat16Bits.fromFloat((float) value);
            case INT32 -> {
                long integral = Math.round(value);
                if (Math.abs(value - integral) > 1e-9) {
                    throw new UnsupportedOperationException("Non-integral write into INT32 storage is not supported.");
                }
                ((Int32Storage) storage).getIntArray()[offset] = (int) integral;
            }
            case INT64 -> {
                long integral = Math.round(value);
                if (Math.abs(value - integral) > 1e-9) {
                    throw new UnsupportedOperationException("Non-integral write into INT64 storage is not supported.");
                }
                ((Int64Storage) storage).getLongArray()[offset] = integral;
            }
            case BOOL -> throw new UnsupportedOperationException("Numeric write into BOOL storage is not supported.");
        }
    }

    public static long version(TensorStorage storage) {
        return storage == null ? 0L : storage.version();
    }

    public static void markModified(TensorStorage storage) {
        if (storage != null) {
            storage.markModified();
        }
    }

    public static float[] float32Data(TensorStorage storage) {
        float[] data = float32DataOrNull(storage);
        if (data == null) {
            throw unsupportedArrayStorage(DataType.FLOAT32, storage);
        }
        return data;
    }

    public static double[] float64Data(TensorStorage storage) {
        double[] data = float64DataOrNull(storage);
        if (data == null) {
            throw unsupportedArrayStorage(DataType.FLOAT64, storage);
        }
        return data;
    }

    public static short[] bfloat16Data(TensorStorage storage) {
        short[] data = bfloat16DataOrNull(storage);
        if (data == null) {
            throw unsupportedArrayStorage(DataType.BFLOAT16, storage);
        }
        return data;
    }

    public static int[] int32Data(TensorStorage storage) {
        int[] data = int32DataOrNull(storage);
        if (data == null) {
            throw unsupportedArrayStorage(DataType.INT32, storage);
        }
        return data;
    }

    public static long[] int64Data(TensorStorage storage) {
        long[] data = int64DataOrNull(storage);
        if (data == null) {
            throw unsupportedArrayStorage(DataType.INT64, storage);
        }
        return data;
    }

    public static byte[] boolData(TensorStorage storage) {
        byte[] data = boolDataOrNull(storage);
        if (data == null) {
            throw unsupportedArrayStorage(DataType.BOOL, storage);
        }
        return data;
    }

    public static float[] float32DataOrNull(TensorStorage storage) {
        return storage instanceof Float32Storage s ? s.getFloatArray() : null;
    }

    public static double[] float64DataOrNull(TensorStorage storage) {
        return storage instanceof Float64Storage s ? s.getDoubleArray() : null;
    }

    public static short[] bfloat16DataOrNull(TensorStorage storage) {
        return storage instanceof BFloat16Storage s ? s.getShortArray() : null;
    }

    public static int[] int32DataOrNull(TensorStorage storage) {
        return storage instanceof Int32Storage s ? s.getIntArray() : null;
    }

    public static long[] int64DataOrNull(TensorStorage storage) {
        return storage instanceof Int64Storage s ? s.getLongArray() : null;
    }

    public static byte[] boolDataOrNull(TensorStorage storage) {
        return storage instanceof BoolStorage s ? s.getByteArray() : null;
    }

    private static UnsupportedOperationException unsupportedArrayStorage(DataType expected, TensorStorage storage) {
        String actual = storage == null ? "null" : storage.getClass().getSimpleName() + "/" + storage.getType();
        return new UnsupportedOperationException(expected + " array storage required, actual=" + actual);
    }

    public static int logicalFlatIndexToStorageOffset(TensorMetadata metadata, int logicalIndex) {
        return metadata.storageOffsetForLogicalFlatIndex(logicalIndex);
    }

    private static void requireSource(Object source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
    }

    private static DataType normalizeDataType(TensorMetadata metadata) {
        if (metadata.getDataType() != null) {
            return metadata.getDataType();
        }
        metadata.setDataType(TensorMetadata.DEFAULT_DATA_TYPE);
        return metadata.getDataType();
    }
}
