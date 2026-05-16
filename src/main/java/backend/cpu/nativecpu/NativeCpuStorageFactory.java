package backend.cpu.nativecpu;

import tensor.DataType;
import tensor.NativeBFloat16Storage;
import tensor.NativeFloat32Storage;
import tensor.NativeFloat64Storage;
import tensor.NativeTensorStorage;

/**
 * Allocates dtype-specific native CPU tensor storage.
 */
public final class NativeCpuStorageFactory {
    private final NativeCpuAllocator allocator;

    public NativeCpuStorageFactory() {
        this(new NativeCpuAllocator());
    }

    public NativeCpuStorageFactory(NativeCpuAllocator allocator) {
        this.allocator = allocator == null ? new NativeCpuAllocator() : allocator;
    }

    public NativeTensorStorage allocate(DataType dataType, int elements, String label) {
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (elements < 0) {
            throw new IllegalArgumentException("elements cannot be negative: " + elements);
        }
        int elementBytes = elementBytes(dataType);
        NativeCpuAllocation allocation = allocator.allocate(Math.multiplyExact((long) elements, elementBytes), label);
        return switch (dataType) {
            case FLOAT32 -> new NativeFloat32Storage(elements, allocation);
            case FLOAT64 -> new NativeFloat64Storage(elements, allocation);
            case BFLOAT16 -> new NativeBFloat16Storage(elements, allocation);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Native CPU storage MVP supports only FLOAT32, FLOAT64, and BFLOAT16. dtype=" + dataType);
        };
    }

    public static int elementBytes(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> Float.BYTES;
            case FLOAT64 -> Double.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
        };
    }
}
