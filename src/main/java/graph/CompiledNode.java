package graph;

import backend.ComputeBackend;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable compile-time snapshot of a semantic tensor node.
 *
 * The semantic tensor reference remains available as a runtime value/storage handle and for debug
 * introspection, but graph topology and node metadata are captured here so prepared execution does
 * not depend on live mutable Tensor structure.
 */
public final class CompiledNode {
    private final int id;
    private final Tensor semanticTensor;
    private final Tensor sourceTensor;
    private final Operation operation;
    private final ComputeBackend backend;
    private final List<Integer> inputIds;
    private final List<Tensor> inputTensors;
    private final int[] shape;
    private final int[] strides;
    private final int storageOffset;
    private final DataType dataType;
    private final boolean backwardNode;
    private final boolean leaf;
    private final boolean contiguous;
    private final boolean hasStorageOffset;
    private final int flatDataSize;
    private final String label;

    private CompiledNode(
            int id,
            Tensor semanticTensor,
            Tensor sourceTensor,
            Operation operation,
            ComputeBackend backend,
            List<Integer> inputIds,
            List<Tensor> inputTensors,
            int[] shape,
            int[] strides,
            int storageOffset,
            DataType dataType,
            boolean backwardNode,
            boolean leaf,
            boolean contiguous,
            boolean hasStorageOffset,
            int flatDataSize,
            String label
    ) {
        this.id = id;
        this.semanticTensor = Objects.requireNonNull(semanticTensor, "semanticTensor cannot be null");
        this.sourceTensor = Objects.requireNonNull(sourceTensor, "sourceTensor cannot be null");
        this.operation = operation;
        this.backend = Objects.requireNonNull(backend, "backend cannot be null");
        this.inputIds = List.copyOf(inputIds == null ? List.of() : inputIds);
        this.inputTensors = List.copyOf(inputTensors == null ? List.of() : inputTensors);
        this.shape = shape == null ? new int[0] : shape.clone();
        this.strides = strides == null ? new int[0] : strides.clone();
        this.storageOffset = storageOffset;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.backwardNode = backwardNode;
        this.leaf = leaf;
        this.contiguous = contiguous;
        this.hasStorageOffset = hasStorageOffset;
        this.flatDataSize = flatDataSize;
        this.label = label == null ? "" : label;
    }

    public static List<CompiledNode> snapshot(List<Tensor> orderedGraph) {
        return snapshot(orderedGraph, Map.of());
    }

    public static List<CompiledNode> snapshot(List<Tensor> orderedGraph, Map<Tensor, Tensor> sourceTensors) {
        if (orderedGraph == null || orderedGraph.isEmpty()) {
            return List.of();
        }
        sourceTensors = sourceTensors == null ? Map.of() : Map.copyOf(sourceTensors);
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < orderedGraph.size(); i++) {
            ids.put(orderedGraph.get(i), i);
        }
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
            out.add(new CompiledNode(
                    i,
                    tensor,
                    sourceTensors.getOrDefault(tensor, tensor),
                    tensor.getOperation(),
                    tensor.resolveBackend(),
                    inputIds,
                    inputs == null ? List.of() : inputs,
                    tensor.getShapeUnsafe(),
                    tensor.getStridesUnsafe(),
                    tensor.getStorageOffsetUnsafe(),
                    tensor.getDataType(),
                    tensor.isBackward(),
                    tensor.getOperation() == null,
                    tensor.isContiguous(),
                    tensor.hasStorageOffset(),
                    tensor.getFlatDataSize(),
                    tensor.getLabel()
            ));
        }
        return List.copyOf(out);
    }

    public int id() {
        return id;
    }

    public Tensor semanticTensor() {
        return semanticTensor;
    }

    public Tensor sourceTensor() {
        return sourceTensor;
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

    public List<Tensor> inputTensors() {
        return inputTensors;
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

    public boolean contiguous() {
        return contiguous;
    }

    public boolean hasStorageOffset() {
        return hasStorageOffset;
    }

    public int flatDataSize() {
        return flatDataSize;
    }

    public String label() {
        return label;
    }
}
