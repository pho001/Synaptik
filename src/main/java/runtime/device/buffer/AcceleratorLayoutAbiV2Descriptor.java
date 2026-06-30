package runtime.device.buffer;

import tensor.DataType;

import java.util.Objects;

/**
 * Backend-neutral metadata passed to native layout ABI v2 capability checks.
 */
public record AcceleratorLayoutAbiV2Descriptor(
        String backendId,
        int nodeId,
        DataType dataType,
        int rank,
        int[] shape,
        int[] strides,
        int storageOffset,
        long logicalElementCount,
        long logicalByteLength,
        long physicalByteSpan,
        AcceleratorBufferAccessMode accessMode,
        AcceleratorBufferLayoutClass layoutClass,
        String nativeHandleIdentity
) {
    public AcceleratorLayoutAbiV2Descriptor {
        backendId = requireNonBlank(backendId, "backendId");
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        Objects.requireNonNull(accessMode, "accessMode cannot be null");
        Objects.requireNonNull(layoutClass, "layoutClass cannot be null");
        nativeHandleIdentity = requireNonBlank(nativeHandleIdentity, "nativeHandleIdentity");
        if (rank <= 0) {
            throw new IllegalArgumentException("rank must be positive");
        }
        if (shape.length != rank || strides.length != rank) {
            throw new IllegalArgumentException("rank must match shape and strides length");
        }
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        if (logicalElementCount < 0) {
            throw new IllegalArgumentException("logicalElementCount cannot be negative");
        }
        long expectedByteLength = AcceleratorBufferLayout.byteLength(dataType, logicalElementCount);
        if (logicalByteLength != expectedByteLength) {
            throw new IllegalArgumentException("logicalByteLength " + logicalByteLength
                    + " does not match dtype/element count byte length " + expectedByteLength);
        }
        for (int stride : strides) {
            if (stride < 0) {
                throw new IllegalArgumentException("negative stride is not supported by layout ABI v2 metadata");
            }
        }
        long expectedPhysicalSpan = physicalByteSpan(dataType, shape, strides, storageOffset);
        if (physicalByteSpan != expectedPhysicalSpan) {
            throw new IllegalArgumentException("physicalByteSpan " + physicalByteSpan
                    + " does not match layout metadata physical span " + expectedPhysicalSpan);
        }
        shape = shape.clone();
        strides = strides.clone();
    }

    public static AcceleratorLayoutAbiV2Descriptor fromBinding(DeviceBufferBinding binding) {
        Objects.requireNonNull(binding, "binding cannot be null");
        AcceleratorBufferLayout layout = Objects.requireNonNull(binding.layout(), "binding layout cannot be null");
        int[] shape = layout.shape();
        int[] strides = layout.strides();
        return new AcceleratorLayoutAbiV2Descriptor(
                binding.backendId(),
                binding.nodeId(),
                layout.dataType(),
                shape.length,
                shape,
                strides,
                layout.storageOffset(),
                layout.logicalElementCount(),
                layout.logicalByteLength(),
                physicalByteSpan(layout),
                binding.accessMode(),
                layout.layoutClass(),
                binding.nativeHandleIdentity()
        );
    }

    public static long physicalByteSpan(AcceleratorBufferLayout layout) {
        Objects.requireNonNull(layout, "layout cannot be null");
        return physicalByteSpan(layout.dataType(), layout.shape(), layout.strides(), layout.storageOffset());
    }

    public static long physicalByteSpan(DataType dataType, int[] shape, int[] strides, int storageOffset) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length");
        }
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        long maxElementOffset = storageOffset;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0) {
                throw new IllegalArgumentException("shape dimensions cannot be negative");
            }
            if (shape[i] == 0) {
                return 0L;
            }
            if (strides[i] < 0) {
                throw new IllegalArgumentException("negative stride is not supported by layout ABI v2 metadata");
            }
            long spanForDimension = Math.multiplyExact((long) shape[i] - 1L, strides[i]);
            maxElementOffset = Math.addExact(maxElementOffset, spanForDimension);
        }
        long physicalElements = Math.addExact(maxElementOffset, 1L);
        return Math.multiplyExact(physicalElements, AcceleratorBufferLayout.bytesPerElement(dataType));
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    private static String requireNonBlank(String value, String label) {
        Objects.requireNonNull(value, label + " cannot be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(label + " cannot be blank");
        }
        return value;
    }
}
