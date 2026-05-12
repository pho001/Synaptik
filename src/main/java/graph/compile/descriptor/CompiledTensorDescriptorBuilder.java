package graph.compile.descriptor;

import graph.CompiledNode;
import operations.Operation;
import tensor.DataType;
import tensor.TensorMetadata;

import java.util.List;
import java.util.Objects;

/**
 * Builds descriptor indexes from immutable compiled node snapshots.
 */
public final class CompiledTensorDescriptorBuilder {
    private CompiledTensorDescriptorBuilder() {
    }

    public static CompiledTensorDescriptorIndex build(List<CompiledNode> nodes) {
        if (nodes == null || nodes.isEmpty()) {
            return CompiledTensorDescriptorIndex.empty();
        }
        return new CompiledTensorDescriptorIndex(nodes.stream()
                .map(CompiledTensorDescriptorBuilder::fromNode)
                .toList());
    }

    public static CompiledTensorDescriptor fromNode(CompiledNode node) {
        Objects.requireNonNull(node, "node cannot be null");
        int[] shape = node.shape();
        int[] strides = node.strides();
        long logicalElementCount = product(shape);
        long physicalSpan = physicalSpan(shape, strides, node.storageOffset());
        DataType dataType = node.dataType();
        boolean hasZeroStride = hasZeroStride(strides);
        LayoutClass layoutClass = classify(shape, strides, node.storageOffset(), logicalElementCount, node.contiguous(), hasZeroStride);
        Operation operation = node.operation();
        return new CompiledTensorDescriptor(
                node.id(),
                operation == null ? null : operation.opType(),
                dataType,
                shape,
                shape.length,
                strides,
                node.storageOffset(),
                logicalElementCount,
                physicalSpan,
                Math.multiplyExact(logicalElementCount, bytesPerElement(dataType)),
                Math.multiplyExact(physicalSpan, bytesPerElement(dataType)),
                layoutClass,
                node.contiguous(),
                node.hasStorageOffset(),
                hasZeroStride,
                hasZeroStride,
                node.leaf(),
                node.backwardNode(),
                node.requiresGrad(),
                node.trainableParameter(),
                node.inputIds()
        );
    }

    private static LayoutClass classify(
            int[] shape,
            int[] strides,
            int storageOffset,
            long logicalElementCount,
            boolean contiguous,
            boolean hasZeroStride
    ) {
        if (shape.length != strides.length || hasNegativeStride(strides) || product(shape) != logicalElementCount) {
            return LayoutClass.UNKNOWN_OR_COMPLEX;
        }
        if (hasZeroStride) {
            return LayoutClass.BROADCAST_ZERO_STRIDE;
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (java.util.Arrays.equals(strides, denseStrides) || contiguous) {
            return storageOffset == 0 ? LayoutClass.DENSE_CONTIGUOUS : LayoutClass.DENSE_WITH_OFFSET;
        }
        return LayoutClass.STRIDED_VIEW;
    }

    private static long product(int[] shape) {
        if (shape == null || shape.length == 0) {
            return 1L;
        }
        long product = 1L;
        for (int dimension : shape) {
            product = Math.multiplyExact(product, dimension);
        }
        return product;
    }

    private static long physicalSpan(int[] shape, int[] strides, int storageOffset) {
        if (shape == null || shape.length == 0) {
            return 1L;
        }
        long maxOffset = storageOffset;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == 0) {
                return 0L;
            }
            long stride = Math.abs((long) strides[i]);
            maxOffset = Math.addExact(maxOffset, Math.multiplyExact((long) shape[i] - 1L, stride));
        }
        return Math.addExact(maxOffset, 1L);
    }

    private static boolean hasZeroStride(int[] strides) {
        if (strides == null) {
            return false;
        }
        for (int stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasNegativeStride(int[] strides) {
        if (strides == null) {
            return false;
        }
        for (int stride : strides) {
            if (stride < 0) {
                return true;
            }
        }
        return false;
    }

    private static int bytesPerElement(DataType dataType) {
        return switch (Objects.requireNonNull(dataType, "dataType cannot be null")) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case INT32 -> Integer.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }
}
