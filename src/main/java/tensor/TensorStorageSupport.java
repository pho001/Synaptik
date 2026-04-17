package tensor;

import backend.kernels.cpu.CpuDTypeOps;

final class TensorStorageSupport {
    private TensorStorageSupport() {
    }

    static TensorStorage emptyStorage(TensorMetadata metadata) {
        return emptyStorage(metadata, metadata.getFlatSize());
    }

    static TensorStorage emptyStorage(TensorMetadata metadata, int size) {
        DataType type = normalizeDataType(metadata);
        if (type == DataType.FLOAT64) {
            return new Float64Storage(size);
        }
        return switch (type) {
            case BOOL -> new BoolStorage(size);
            case BFLOAT16 -> new BFloat16Storage(size);
            case FLOAT32 -> new Float32Storage(size);
            case INT32 -> new Int32Storage(size);
            case FLOAT64 -> throw new IllegalStateException("Unexpected dtype branch");
        };
    }

    static TensorStorage fromDoubleArray(TensorMetadata metadata, double[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type == DataType.BOOL || type == DataType.INT32) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
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
            converted[i] = CpuDTypeOps.toBFloat16Bits((float) source[i]);
        }
        return new BFloat16Storage(converted);
    }

    static TensorStorage fromFloatArray(TensorMetadata metadata, float[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type == DataType.BOOL || type == DataType.INT32) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
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
                    converted[i] = CpuDTypeOps.toBFloat16Bits(source[i]);
                }
                yield new BFloat16Storage(converted);
            }
            case BOOL, INT32 -> throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
        };
    }

    static TensorStorage fromBFloat16Array(TensorMetadata metadata, short[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type == DataType.BOOL || type == DataType.INT32) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
        }
        int size = source.length;
        return switch (type) {
            case BFLOAT16 -> new BFloat16Storage(source);
            case FLOAT32 -> {
                float[] converted = new float[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = CpuDTypeOps.fromBFloat16Bits(source[i]);
                }
                yield new Float32Storage(converted);
            }
            case FLOAT64 -> {
                double[] converted = new double[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = CpuDTypeOps.fromBFloat16Bits(source[i]);
                }
                yield new Float64Storage(converted);
            }
            case BOOL, INT32 -> throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
        };
    }

    static TensorStorage fromBoolArray(TensorMetadata metadata, byte[] source) {
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

    static TensorStorage fromIntArray(TensorMetadata metadata, int[] source) {
        requireSource(source);
        DataType type = normalizeDataType(metadata);
        if (type != DataType.INT32) {
            throw new UnsupportedOperationException("Implicit INT32 -> other dtype conversion is not supported.");
        }
        return new Int32Storage(source);
    }

    static void validateInputLength(int actual, int expected, String sourceName) {
        if (actual != expected) {
            throw new IllegalArgumentException(sourceName + " length mismatch. expected=" + expected + ", actual=" + actual);
        }
    }

    static double getByStorageOffset(TensorStorage storage, int storageSize, int offset) {
        if (storage == null) {
            throw new IllegalStateException("Tensor storage is not initialized.");
        }
        if (offset < 0 || offset >= storageSize) {
            throw new IndexOutOfBoundsException("Storage offset out of bounds.");
        }
        return switch (storage.getType()) {
            case FLOAT64 -> ((Float64Storage) storage).getDoubleArray()[offset];
            case FLOAT32 -> ((Float32Storage) storage).getFloatArray()[offset];
            case BFLOAT16 -> CpuDTypeOps.fromBFloat16Bits(((BFloat16Storage) storage).getShortArray()[offset]);
            case INT32 -> ((Int32Storage) storage).getIntArray()[offset];
            case BOOL -> ((BoolStorage) storage).getByteArray()[offset] == 0 ? 0.0d : 1.0d;
        };
    }

    static void setByStorageOffset(TensorStorage storage, int storageSize, int offset, double value) {
        if (storage == null) {
            throw new IllegalStateException("Tensor storage is not initialized.");
        }
        if (offset < 0 || offset >= storageSize) {
            throw new IndexOutOfBoundsException("Storage offset out of bounds.");
        }
        switch (storage.getType()) {
            case FLOAT64 -> ((Float64Storage) storage).getDoubleArray()[offset] = value;
            case FLOAT32 -> ((Float32Storage) storage).getFloatArray()[offset] = (float) value;
            case BFLOAT16 -> ((BFloat16Storage) storage).getShortArray()[offset] = CpuDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> {
                long integral = Math.round(value);
                if (Math.abs(value - integral) > 1e-9) {
                    throw new UnsupportedOperationException("Non-integral write into INT32 storage is not supported.");
                }
                ((Int32Storage) storage).getIntArray()[offset] = (int) integral;
            }
            case BOOL -> throw new UnsupportedOperationException("Numeric write into BOOL storage is not supported.");
        }
    }

    static long version(TensorStorage storage) {
        return storage == null ? 0L : storage.version();
    }

    static void markModified(TensorStorage storage) {
        if (storage != null) {
            storage.markModified();
        }
    }

    static float[] float32Data(TensorStorage storage) {
        return storage instanceof Float32Storage s ? s.getFloatArray() : null;
    }

    static double[] float64Data(TensorStorage storage) {
        return storage instanceof Float64Storage s ? s.getDoubleArray() : null;
    }

    static short[] bfloat16Data(TensorStorage storage) {
        return storage instanceof BFloat16Storage s ? s.getShortArray() : null;
    }

    static int[] int32Data(TensorStorage storage) {
        return storage instanceof Int32Storage s ? s.getIntArray() : null;
    }

    static byte[] boolData(TensorStorage storage) {
        return storage instanceof BoolStorage s ? s.getByteArray() : null;
    }

    static int logicalFlatIndexToStorageOffset(TensorMetadata metadata, int logicalIndex) {
        int[] shape = metadata.shapeRef();
        int[] strides = metadata.stridesRef();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        int rem = logicalIndex;
        int offset = metadata.getStorageOffset();
        for (int dim = 0; dim < shape.length; dim++) {
            int coord = rem / denseStrides[dim];
            rem %= denseStrides[dim];
            offset += coord * strides[dim];
        }
        return offset;
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
