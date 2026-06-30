package backend.cpu1.storage;

import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import tensor.TensorMetadata;
import tensor.layout.BroadcastPlan;
import tensor.layout.BroadcastPlanner;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepare-time storage access metadata for cpu1 routing decisions.
 *
 * <p>This plan deliberately carries layout facts only. Runtime storage handles and typed
 * accessors belong in executable views and kernels.</p>
 */
public record Cpu1StorageAccessPlan(
        Cpu1StorageAccessKind kind,
        int[] shape,
        int[] strides,
        int storageOffset,
        long elementCount,
        String rejectionReason
) {
    public Cpu1StorageAccessPlan {
        Objects.requireNonNull(kind, "kind cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides length must match");
        }
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount cannot be negative");
        }
        for (int dimension : shape) {
            if (dimension < 0) {
                throw new IllegalArgumentException("shape dimensions cannot be negative");
            }
        }
        if (kind != Cpu1StorageAccessKind.UNSUPPORTED && hasNegativeStride(strides)) {
            throw new IllegalArgumentException("negative strides are unsupported");
        }
        if (kind == Cpu1StorageAccessKind.UNSUPPORTED
                && (rejectionReason == null || rejectionReason.isBlank())) {
            throw new IllegalArgumentException("unsupported access plans require a rejection reason");
        }
        shape = shape.clone();
        strides = strides.clone();
    }

    public static Cpu1StorageAccessPlan fromDescriptor(CompiledTensorDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor cannot be null");
        return classify(
                descriptor.shape(),
                descriptor.strides(),
                descriptor.storageOffset(),
                descriptor.logicalElementCount(),
                descriptor.contiguous()
        );
    }

    public static Cpu1StorageAccessPlan fromNode(CompiledNode node) {
        Objects.requireNonNull(node, "node cannot be null");
        return classify(
                node.shape(),
                node.strides(),
                node.storageOffset(),
                node.flatDataSize(),
                node.contiguous()
        );
    }

    public static Cpu1StorageAccessPlan forBroadcastedLogicalShape(
            CompiledTensorDescriptor descriptor,
            int[] logicalShape
    ) {
        Objects.requireNonNull(descriptor, "descriptor cannot be null");
        Objects.requireNonNull(logicalShape, "logicalShape cannot be null");
        int[] logicalShapeCopy = logicalShape.clone();
        int[] denseLogicalStrides = TensorMetadata.computeStrides(logicalShapeCopy);
        long logicalElementCount = product(logicalShapeCopy);
        try {
            BroadcastPlan broadcastPlan = BroadcastPlanner.plan(
                    descriptor.shape(),
                    descriptor.strides(),
                    logicalShapeCopy,
                    denseLogicalStrides
            );
            if (!Arrays.equals(broadcastPlan.outShape(), logicalShapeCopy)) {
                return unsupported(
                        logicalShapeCopy,
                        denseLogicalStrides,
                        descriptor.storageOffset(),
                        logicalElementCount,
                        "descriptor shape is not broadcast-compatible with logical shape"
                );
            }
            return classify(
                    logicalShapeCopy,
                    broadcastPlan.aEffStrides(),
                    descriptor.storageOffset(),
                    logicalElementCount,
                    false
            );
        } catch (IllegalArgumentException e) {
            return unsupported(
                    logicalShapeCopy,
                    denseLogicalStrides,
                    descriptor.storageOffset(),
                    logicalElementCount,
                    "descriptor shape is not broadcast-compatible with logical shape: " + e.getMessage()
            );
        }
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    private static Cpu1StorageAccessPlan classify(
            int[] shape,
            int[] strides,
            int storageOffset,
            long elementCount,
            boolean contiguous
    ) {
        validateMetadata(shape, strides, storageOffset, elementCount);
        int[] shapeCopy = shape.clone();
        int[] stridesCopy = strides.clone();
        if (hasNegativeStride(stridesCopy)) {
            return unsupported(shapeCopy, stridesCopy, storageOffset, elementCount, "negative strides are unsupported");
        }
        long expectedElementCount = product(shapeCopy);
        if (expectedElementCount != elementCount) {
            return unsupported(
                    shapeCopy,
                    stridesCopy,
                    storageOffset,
                    elementCount,
                    "elementCount does not match shape product"
            );
        }
        if (hasZeroStride(stridesCopy)) {
            return new Cpu1StorageAccessPlan(
                    Cpu1StorageAccessKind.BROADCAST,
                    shapeCopy,
                    stridesCopy,
                    storageOffset,
                    elementCount,
                    null
            );
        }
        boolean denseContiguous = contiguous || hasDenseContiguousStrides(shapeCopy, stridesCopy);
        if (denseContiguous) {
            Cpu1StorageAccessKind kind = storageOffset == 0
                    ? Cpu1StorageAccessKind.DENSE_CONTIGUOUS
                    : Cpu1StorageAccessKind.DENSE_WITH_OFFSET;
            return new Cpu1StorageAccessPlan(kind, shapeCopy, stridesCopy, storageOffset, elementCount, null);
        }
        return new Cpu1StorageAccessPlan(
                Cpu1StorageAccessKind.STRIDED,
                shapeCopy,
                stridesCopy,
                storageOffset,
                elementCount,
                null
        );
    }

    private static Cpu1StorageAccessPlan unsupported(
            int[] shape,
            int[] strides,
            int storageOffset,
            long elementCount,
            String rejectionReason
    ) {
        return new Cpu1StorageAccessPlan(
                Cpu1StorageAccessKind.UNSUPPORTED,
                shape,
                strides,
                storageOffset,
                elementCount,
                rejectionReason
        );
    }

    private static void validateMetadata(int[] shape, int[] strides, int storageOffset, long elementCount) {
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides length must match");
        }
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount cannot be negative");
        }
        for (int dimension : shape) {
            if (dimension < 0) {
                throw new IllegalArgumentException("shape dimensions cannot be negative");
            }
        }
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

    private static boolean hasDenseContiguousStrides(int[] shape, int[] strides) {
        long expectedStride = 1L;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (strides[dim] != expectedStride) {
                return false;
            }
            expectedStride = Math.multiplyExact(expectedStride, shape[dim]);
            if (expectedStride > Integer.MAX_VALUE && dim > 0) {
                return false;
            }
        }
        return true;
    }

    private static long product(int[] shape) {
        long product = 1L;
        for (int dimension : shape) {
            product = Math.multiplyExact(product, dimension);
        }
        return product;
    }
}
