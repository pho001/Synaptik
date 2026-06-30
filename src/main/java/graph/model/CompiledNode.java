package graph.model;

import backend.contract.ComputeBackend;
import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Immutable compile-time snapshot of a tensor graph node.
 *
 * <p>Compile topology and node metadata are captured as values so prepare/lowering does not depend on the mutable
 * Tensor graph. User-visible publication bindings live in {@code PublicationPlan}, not in node snapshots. Operation
 * instances referenced by compiled nodes are compile metadata and must be treated as immutable.
 */
public final class CompiledNode {
    private final int id;
    private final Operation operation;
    private final ComputeBackend backend;
    private final List<Integer> inputIds;
    private final int storageOwnerId;
    private final int[] shape;
    private final int[] strides;
    private final int storageOffset;
    private final DataType dataType;
    private final boolean backwardNode;
    private final boolean leaf;
    private final boolean requiresGrad;
    private final boolean trainableParameter;
    private final boolean contiguous;
    private final boolean hasStorageOffset;
    private final boolean gradientTarget;
    private final int flatDataSize;
    private final String label;
    private final CompiledTensorDataSnapshot staticDataSnapshot;

    private CompiledNode(
            int id,
            Operation operation,
            ComputeBackend backend,
            List<Integer> inputIds,
            int storageOwnerId,
            int[] shape,
            int[] strides,
            int storageOffset,
            DataType dataType,
            boolean backwardNode,
            boolean leaf,
            boolean requiresGrad,
            boolean trainableParameter,
            boolean contiguous,
            boolean hasStorageOffset,
            boolean gradientTarget,
            int flatDataSize,
            String label,
            CompiledTensorDataSnapshot staticDataSnapshot
    ) {
        this.id = id;
        this.operation = operation;
        this.backend = Objects.requireNonNull(backend, "backend cannot be null");
        this.inputIds = List.copyOf(inputIds == null ? List.of() : inputIds);
        this.storageOwnerId = storageOwnerId;
        this.shape = shape == null ? new int[0] : shape.clone();
        this.strides = strides == null ? new int[0] : strides.clone();
        this.storageOffset = storageOffset;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.backwardNode = backwardNode;
        this.leaf = leaf;
        this.requiresGrad = requiresGrad;
        this.trainableParameter = trainableParameter;
        this.contiguous = contiguous;
        this.hasStorageOffset = hasStorageOffset;
        this.gradientTarget = gradientTarget;
        this.flatDataSize = flatDataSize;
        this.label = label == null ? "" : label;
        this.staticDataSnapshot = staticDataSnapshot == null ? CompiledTensorDataSnapshot.EMPTY : staticDataSnapshot;
    }

    /**
     * Creates an immutable model value at the compile-time Tensor-to-model snapshot boundary.
     *
     * <p>Callers outside compile snapshotting should consume existing compiled nodes rather than constructing them.
     */
    public static CompiledNode compiledSnapshot(
            int id,
            Operation operation,
            ComputeBackend backend,
            List<Integer> inputIds,
            int storageOwnerId,
            int[] shape,
            int[] strides,
            int storageOffset,
            DataType dataType,
            boolean backwardNode,
            boolean leaf,
            boolean requiresGrad,
            boolean trainableParameter,
            boolean contiguous,
            boolean hasStorageOffset,
            boolean gradientTarget,
            int flatDataSize,
            String label,
            CompiledTensorDataSnapshot staticDataSnapshot
    ) {
        return new CompiledNode(
                id,
                operation,
                backend,
                inputIds,
                storageOwnerId,
                shape,
                strides,
                storageOffset,
                dataType,
                backwardNode,
                leaf,
                requiresGrad,
                trainableParameter,
                contiguous,
                hasStorageOffset,
                gradientTarget,
                flatDataSize,
                label,
                staticDataSnapshot
        );
    }

    public int id() {
        return id;
    }

    public Operation operation() {
        return operation;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public List<Integer> inputIds() {
        return inputIds;
    }

    public int storageOwnerId() {
        return storageOwnerId;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int[] strides() {
        return strides.clone();
    }

    public int storageOffset() {
        return storageOffset;
    }

    public DataType dataType() {
        return dataType;
    }

    public boolean backwardNode() {
        return backwardNode;
    }

    public boolean leaf() {
        return leaf;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public boolean trainableParameter() {
        return trainableParameter;
    }

    public boolean contiguous() {
        return contiguous;
    }

    public boolean hasStorageOffset() {
        return hasStorageOffset;
    }

    public boolean gradientTarget() {
        return gradientTarget;
    }

    public int flatDataSize() {
        return flatDataSize;
    }

    public String label() {
        return label;
    }

    public CompiledTensorDataSnapshot staticDataSnapshot() {
        return staticDataSnapshot;
    }
}
