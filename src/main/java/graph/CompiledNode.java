package graph;

import backend.ComputeBackend;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable compile-time snapshot of a tensor graph node.
 *
 * <p>The user-visible publication tensor reference remains available for runtime input seeding, output and
 * gradient publication, and debug introspection. Compile topology and node metadata are captured as values so
 * prepare/lowering does not depend on the mutable Tensor graph. Operation instances referenced by compiled nodes
 * are compile metadata and must be treated as immutable.
 */
public final class CompiledNode {
    private final int id;
    private final Tensor publicationTensor;
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
            Tensor publicationTensor,
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
        this.publicationTensor = Objects.requireNonNull(publicationTensor, "publicationTensor cannot be null");
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
     * Captures compiled node snapshots for an ordered graph.
     *
     * @param orderedGraph tensors in topological order
     * @return immutable compiled node snapshots
     */
    public static List<CompiledNode> snapshot(List<Tensor> orderedGraph) {
        return snapshot(orderedGraph, Map.of());
    }

    /**
     * Captures compiled node snapshots with optional publication tensor remapping.
     *
     * @param orderedGraph tensors in topological order
     * @param publicationTensors mapping from compiled graph tensors to user-visible publication tensors
     * @return immutable compiled node snapshots
     */
    public static List<CompiledNode> snapshot(List<Tensor> orderedGraph, Map<Tensor, Tensor> publicationTensors) {
        if (orderedGraph == null || orderedGraph.isEmpty()) {
            return List.of();
        }
        publicationTensors = publicationTensors == null ? Map.of() : Map.copyOf(publicationTensors);
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < orderedGraph.size(); i++) {
            ids.put(orderedGraph.get(i), i);
        }
        Set<Tensor> gradientTargets = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor tensor : orderedGraph) {
            Tensor gradient = tensor.getGradient();
            if (gradient != null && ids.containsKey(gradient)) {
                gradientTargets.add(gradient);
            }
        }
        IdentityHashMap<Tensor, Integer> storageOwnerIds = new IdentityHashMap<>();
        List<CompiledNode> out = new ArrayList<>(orderedGraph.size());
        for (int i = 0; i < orderedGraph.size(); i++) {
            Tensor tensor = orderedGraph.get(i);
            List<Tensor> inputs = tensor.getPrevTensors();
            List<Integer> inputIds = new ArrayList<>(inputs == null ? 0 : inputs.size());
            if (inputs != null) {
                for (Tensor input : inputs) {
                    Integer inputId = ids.get(input);
                    if (inputId == null) {
                        throw new IllegalStateException("Compiled node input is missing from ordered graph: " + tensor.getLabel());
                    }
                    inputIds.add(inputId);
                }
            }
            int storageOwnerId = resolveStorageOwnerId(i, tensor, inputs, storageOwnerIds);
            storageOwnerIds.put(tensor, storageOwnerId);
            out.add(new CompiledNode(
                    i,
                    publicationTensors.getOrDefault(tensor, tensor),
                    tensor.getOperation(),
                    TensorInternalAccess.backendIntent(tensor),
                    inputIds,
                    storageOwnerId,
                    tensor.getShapeUnsafe(),
                    tensor.getStridesUnsafe(),
                    tensor.getStorageOffsetUnsafe(),
                    tensor.getDataType(),
                    tensor.isBackward(),
                    tensor.getOperation() == null,
                    tensor.getRequiresGrad(),
                    tensor.isTrainableParameter(),
                    tensor.isContiguous(),
                    tensor.hasStorageOffset(),
                    gradientTargets.contains(tensor),
                    tensor.getFlatDataSize(),
                    tensor.getLabel(),
                    CompiledTensorDataSnapshot.captureStaticLeaf(tensor)
            ));
        }
        return List.copyOf(out);
    }

    private static int resolveStorageOwnerId(
            int nodeId,
            Tensor tensor,
            List<Tensor> inputs,
            IdentityHashMap<Tensor, Integer> storageOwnerIds
    ) {
        if (!AliasViewPolicy.aliasesInput0AtRuntime(tensor) || inputs == null || inputs.isEmpty()) {
            return nodeId;
        }
        Tensor input0 = inputs.getFirst();
        return storageOwnerIds.getOrDefault(input0, nodeId);
    }

    public int id() {
        return id;
    }

    public Tensor publicationTensor() {
        return publicationTensor;
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
