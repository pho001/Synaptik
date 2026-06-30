package planning.descriptor;

import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Immutable compile-time tensor facts derived from {@link graph.model.CompiledNode}.
 *
 * <p>This descriptor is the stable source for prepare-time capability, layout, lowering, and
 * trace decisions. Public {@code Tensor} objects remain logical/runtime objects; this type is
 * rebuilt from compiled snapshots and is not mutable runtime state.</p>
 *
 * <p>{@link Operation} objects referenced by compiled nodes and descriptors must be treated as
 * immutable compile metadata.</p>
 */
public record CompiledTensorDescriptor(
        int nodeId,
        Operation.OpType opType,
        DataType dataType,
        int[] shape,
        int rank,
        int[] strides,
        int storageOffset,
        long logicalElementCount,
        long physicalSpan,
        long logicalByteLength,
        long physicalByteSpan,
        LayoutClass layoutClass,
        boolean contiguous,
        boolean hasStorageOffset,
        boolean hasZeroStride,
        boolean broadcastView,
        boolean leaf,
        boolean backwardNode,
        boolean requiresGrad,
        boolean trainableParameter,
        List<Integer> inputIds
) {
    public CompiledTensorDescriptor {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        Objects.requireNonNull(layoutClass, "layoutClass cannot be null");
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        if (rank != shape.length || rank != strides.length) {
            throw new IllegalArgumentException("rank must match shape and strides length");
        }
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        if (logicalElementCount < 0 || physicalSpan < 0 || logicalByteLength < 0 || physicalByteSpan < 0) {
            throw new IllegalArgumentException("descriptor size fields cannot be negative");
        }
        shape = shape.clone();
        strides = strides.clone();
        inputIds = List.copyOf(inputIds == null ? List.of() : inputIds);
    }

    public boolean hasOperation() {
        return opType != null;
    }

    public boolean dense() {
        return layoutClass == LayoutClass.DENSE_CONTIGUOUS || layoutClass == LayoutClass.DENSE_WITH_OFFSET;
    }

    public boolean denseContiguousWithoutOffset() {
        return layoutClass == LayoutClass.DENSE_CONTIGUOUS;
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }
}
