package backend.cpu.nativecpu.layout;

import graph.compile.descriptor.CompiledTensorDescriptor;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Storage-neutral physical tensor view used before lowering to array or native-segment access.
 */
public record TensorPhysicalView(
        int nodeId,
        DataType dataType,
        int[] shape,
        int[] elementStrides,
        int storageOffsetElements,
        long logicalElementCount,
        long logicalByteLength,
        long physicalElementSpan,
        long physicalByteSpan,
        NativeCpuStorageFamily storageFamily,
        NativeCpuLayoutClass layoutClass
) {
    public TensorPhysicalView {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(elementStrides, "elementStrides cannot be null");
        storageFamily = storageFamily == null ? NativeCpuStorageFamily.CPU_ARRAY : storageFamily;
        layoutClass = layoutClass == null ? classify(shape, elementStrides, storageOffsetElements) : layoutClass;
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        if (shape.length != elementStrides.length) {
            throw new IllegalArgumentException("shape and elementStrides must have the same length");
        }
        if (storageOffsetElements < 0) {
            throw new IllegalArgumentException("storageOffsetElements cannot be negative");
        }
        for (int dimension : shape) {
            if (dimension < 0) {
                throw new IllegalArgumentException("shape dimensions cannot be negative");
            }
        }
        for (int stride : elementStrides) {
            if (stride < 0) {
                throw new IllegalArgumentException("negative strides are unsupported for native CPU physical views");
            }
        }
        long expectedLogicalElements = logicalElementCount(shape);
        if (logicalElementCount != expectedLogicalElements) {
            throw new IllegalArgumentException("logicalElementCount " + logicalElementCount
                    + " does not match shape product " + expectedLogicalElements);
        }
        long expectedLogicalBytes = Math.multiplyExact(logicalElementCount, bytesPerElement(dataType));
        if (logicalByteLength != expectedLogicalBytes) {
            throw new IllegalArgumentException("logicalByteLength " + logicalByteLength
                    + " does not match dtype/element count byte length " + expectedLogicalBytes);
        }
        long expectedPhysicalElements = physicalElementSpan(shape, elementStrides, storageOffsetElements);
        if (physicalElementSpan != expectedPhysicalElements) {
            throw new IllegalArgumentException("physicalElementSpan " + physicalElementSpan
                    + " does not match layout span " + expectedPhysicalElements);
        }
        long expectedPhysicalBytes = Math.multiplyExact(physicalElementSpan, bytesPerElement(dataType));
        if (physicalByteSpan != expectedPhysicalBytes) {
            throw new IllegalArgumentException("physicalByteSpan " + physicalByteSpan
                    + " does not match layout byte span " + expectedPhysicalBytes);
        }
        shape = shape.clone();
        elementStrides = elementStrides.clone();
    }

    public static TensorPhysicalView fromDescriptor(
            CompiledTensorDescriptor descriptor,
            NativeCpuStorageFamily storageFamily
    ) {
        Objects.requireNonNull(descriptor, "descriptor cannot be null");
        int[] shape = descriptor.shape();
        int[] strides = descriptor.strides();
        return of(
                descriptor.nodeId(),
                descriptor.dataType(),
                shape,
                strides,
                descriptor.storageOffset(),
                storageFamily
        );
    }

    public static TensorPhysicalView of(
            int nodeId,
            DataType dataType,
            int[] shape,
            int[] elementStrides,
            int storageOffsetElements,
            NativeCpuStorageFamily storageFamily
    ) {
        long logicalElements = logicalElementCount(shape);
        long physicalElements = physicalElementSpan(shape, elementStrides, storageOffsetElements);
        int elementBytes = bytesPerElement(dataType);
        return new TensorPhysicalView(
                nodeId,
                dataType,
                shape,
                elementStrides,
                storageOffsetElements,
                logicalElements,
                Math.multiplyExact(logicalElements, elementBytes),
                physicalElements,
                Math.multiplyExact(physicalElements, elementBytes),
                storageFamily,
                classify(shape, elementStrides, storageOffsetElements)
        );
    }

    public int elementSizeBytes() {
        return bytesPerElement(dataType);
    }

    public boolean denseContiguous() {
        return layoutClass == NativeCpuLayoutClass.DENSE_CONTIGUOUS
                || layoutClass == NativeCpuLayoutClass.OFFSET_CONTIGUOUS;
    }

    public boolean hasStorageOffset() {
        return storageOffsetElements != 0;
    }

    public boolean hasZeroStride() {
        for (int stride : elementStrides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    public long baseByteOffset() {
        return Math.multiplyExact((long) storageOffsetElements, elementSizeBytes());
    }

    public long[] byteStrides() {
        long[] out = new long[elementStrides.length];
        int elementBytes = elementSizeBytes();
        for (int i = 0; i < elementStrides.length; i++) {
            out[i] = Math.multiplyExact((long) elementStrides[i], elementBytes);
        }
        return out;
    }

    public String describe() {
        return "nodeId=" + nodeId
                + ", dtype=" + dataType
                + ", storageFamily=" + storageFamily
                + ", layoutClass=" + layoutClass
                + ", shape=" + Arrays.toString(shape)
                + ", strides=" + Arrays.toString(elementStrides)
                + ", storageOffset=" + storageOffsetElements
                + ", logicalElements=" + logicalElementCount
                + ", physicalByteSpan=" + physicalByteSpan;
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] elementStrides() {
        return elementStrides.clone();
    }

    static NativeCpuLayoutClass classify(int[] shape, int[] strides, int storageOffsetElements) {
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length || storageOffsetElements < 0 || hasNegativeStride(strides)) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT;
        }
        if (hasZeroStride(strides)) {
            if (isLastDimBiasBroadcastRead(shape, strides)) {
                return NativeCpuLayoutClass.LAST_DIM_BIAS_BROADCAST;
            }
            return NativeCpuLayoutClass.BROADCAST_READ_DENSE_WRITE;
        }
        if (Arrays.equals(strides, denseStrides(shape))) {
            return storageOffsetElements == 0
                    ? NativeCpuLayoutClass.DENSE_CONTIGUOUS
                    : NativeCpuLayoutClass.OFFSET_CONTIGUOUS;
        }
        if (isRank2TransposeRead(shape, strides)) {
            return NativeCpuLayoutClass.TRANSPOSE_2D_READ_DENSE_WRITE;
        }
        return NativeCpuLayoutClass.GENERAL_STRIDED_READ_DENSE_WRITE;
    }

    static long logicalElementCount(int[] shape) {
        Objects.requireNonNull(shape, "shape cannot be null");
        if (shape.length == 0) {
            return 1L;
        }
        long product = 1L;
        for (int dimension : shape) {
            if (dimension < 0) {
                throw new IllegalArgumentException("shape dimensions cannot be negative");
            }
            product = Math.multiplyExact(product, dimension);
        }
        return product;
    }

    static long physicalElementSpan(int[] shape, int[] strides, int storageOffsetElements) {
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length");
        }
        if (storageOffsetElements < 0) {
            throw new IllegalArgumentException("storageOffsetElements cannot be negative");
        }
        long maxElementOffset = storageOffsetElements;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0) {
                throw new IllegalArgumentException("shape dimensions cannot be negative");
            }
            if (shape[i] == 0) {
                return 0L;
            }
            if (strides[i] < 0) {
                throw new IllegalArgumentException("negative strides are unsupported for native CPU physical views");
            }
            maxElementOffset = Math.addExact(
                    maxElementOffset,
                    Math.multiplyExact((long) shape[i] - 1L, strides[i])
            );
        }
        return Math.addExact(maxElementOffset, 1L);
    }

    static int bytesPerElement(DataType dataType) {
        return switch (Objects.requireNonNull(dataType, "dataType cannot be null")) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = Math.multiplyExact(stride, shape[i]);
        }
        return strides;
    }

    private static boolean hasZeroStride(int[] strides) {
        for (int stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasNegativeStride(int[] strides) {
        for (int stride : strides) {
            if (stride < 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean isRank2TransposeRead(int[] shape, int[] strides) {
        return shape.length == 2
                && strides.length == 2
                && shape[0] > 0
                && shape[1] > 0
                && strides[0] == 1
                && strides[1] >= shape[0];
    }

    private static boolean isLastDimBiasBroadcastRead(int[] shape, int[] strides) {
        if (shape.length < 2 || strides.length != shape.length || shape[shape.length - 1] <= 0) {
            return false;
        }
        int last = strides.length - 1;
        if (strides[last] != 1) {
            return false;
        }
        for (int i = 0; i < last; i++) {
            if (strides[i] != 0) {
                return false;
            }
        }
        return true;
    }
}
