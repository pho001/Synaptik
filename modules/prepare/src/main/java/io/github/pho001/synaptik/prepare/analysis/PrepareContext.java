package io.github.pho001.synaptik.prepare.analysis;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Provides the complete validated projection for analyzing one planned partition.
 *
 * <p>The projection contains the partition's ordered semantic nodes, projected logical values,
 * matching logical-memory requirements, compile-time logical-splat constants, and one opaque
 * backend input object. It deliberately exposes no Compiler aggregate or implementation type.
 * Every collection is an immutable membership snapshot; list order and map encounter order are
 * deterministic from the supplied containers, while contained immutable references are retained
 * exactly.</p>
 *
 * <p>Construction fails closed unless node order matches the partition, every referenced value is
 * resolved by one unique projected-value entry, every projected value has one descriptor-matching
 * logical requirement, every projected shape is fully static, and each constant is an exact-typed
 * projected graph input. The contract does not separately require every otherwise valid projected
 * value to occur in a node input or output. These checks complete before a context can reach
 * backend analysis. This value does not bind dynamic dimensions, select a route, assign a slot,
 * allocate storage, or contain executable state.</p>
 *
 * @param <I> concrete backend-owned immutable analysis-input role
 * @param partition non-null planned partition retained by exact reference
 * @param nodes non-null ordered nodes to snapshot; elements must be non-null and their IDs must
 *     equal the partition's ordered node IDs
 * @param values non-null ordered projected values to snapshot; elements must be non-null, unique
 *     by value ID, and have fully static descriptor shapes
 * @param memoryRequirements non-null ordered logical requirements to snapshot; elements must be
 *     non-null and provide exactly one descriptor-matching entry for each projected value
 * @param constants non-null map of projected graph-input IDs to exact-typed logical splat values;
 *     keys and values must be non-null, and membership is copied in supplied encounter order
 * @param backendInputs non-null backend-owned immutable input retained opaquely by exact reference
 */
public record PrepareContext<I extends BackendAnalysisInputs>(
        PlannedPartition partition,
        List<CompiledNode> nodes,
        List<GraphValue> values,
        List<LogicalMemoryRequirement> memoryRequirements,
        Map<ValueId, ScalarValue> constants,
        I backendInputs) {
    /**
     * Validates and snapshots one complete partition-analysis projection.
     *
     * <p>Top-level references are validated in component order. List elements and duplicate value
     * identities are then checked in supplied order, followed by partition order, static-shape,
     * reference-closure, logical-requirement, and constant checks. No supplied collection
     * container is retained.</p>
     *
     * @param partition non-null planned partition to retain exactly
     * @param nodes non-null ordered nodes to snapshot
     * @param values non-null ordered projected values to snapshot
     * @param memoryRequirements non-null ordered projected logical requirements to snapshot
     * @param constants non-null projected logical-splat constants to snapshot
     * @param backendInputs non-null immutable backend input to retain opaquely and exactly
     * @throws NullPointerException if a component, list element, constant key, or constant value
     *     is {@code null}; the message identifies the component or indexed entry
     * @throws IllegalArgumentException if node order differs from the partition, a projected value
     *     or requirement ID is duplicated, a descriptor shape is not fully static, a node input or
     *     output is absent from the projected values, requirements do not match projected values
     *     exactly, or a constant is not an exact-typed projected graph input
     */
    public PrepareContext {
        Objects.requireNonNull(partition, "partition");
        Objects.requireNonNull(nodes, "nodes");
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(memoryRequirements, "memoryRequirements");
        Objects.requireNonNull(constants, "constants");
        Objects.requireNonNull(backendInputs, "backendInputs");

        for (int index = 0; index < nodes.size(); index++) {
            Objects.requireNonNull(nodes.get(index), "nodes[" + index + "]");
        }
        nodes = List.copyOf(nodes);

        if (nodes.size() != partition.nodeIds().size()) {
            throw new IllegalArgumentException(
                    "nodes size "
                            + nodes.size()
                            + " does not match partition nodeIds size "
                            + partition.nodeIds().size());
        }
        for (int index = 0; index < nodes.size(); index++) {
            if (!nodes.get(index).id().equals(partition.nodeIds().get(index))) {
                throw new IllegalArgumentException(
                        "nodes["
                                + index
                                + "].id must equal partition.nodeIds["
                                + index
                                + "]: expected "
                                + partition.nodeIds().get(index)
                                + " but was "
                                + nodes.get(index).id());
            }
        }

        var valuesById = new LinkedHashMap<ValueId, GraphValue>();
        for (int index = 0; index < values.size(); index++) {
            GraphValue value = Objects.requireNonNull(values.get(index), "values[" + index + "]");
            if (valuesById.putIfAbsent(value.id(), value) != null) {
                throw new IllegalArgumentException(
                        "values[" + index + "].id duplicates " + value.id());
            }
            if (!value.descriptor().shape().isFullyStatic()) {
                throw new IllegalArgumentException(
                        "values["
                                + index
                                + "].descriptor.shape must be fully static: "
                                + value.descriptor().shape());
            }
        }
        values = List.copyOf(values);

        var requirementsById = new LinkedHashMap<ValueId, LogicalMemoryRequirement>();
        var requirementIndexes = new HashMap<ValueId, Integer>();
        for (int index = 0; index < memoryRequirements.size(); index++) {
            LogicalMemoryRequirement requirement = Objects.requireNonNull(
                    memoryRequirements.get(index), "memoryRequirements[" + index + "]");
            if (requirementsById.putIfAbsent(requirement.valueId(), requirement) != null) {
                throw new IllegalArgumentException(
                        "memoryRequirements["
                                + index
                                + "].valueId duplicates "
                                + requirement.valueId());
            }
            requirementIndexes.put(requirement.valueId(), index);
        }
        memoryRequirements = List.copyOf(memoryRequirements);

        var projectedInputIds = new HashSet<ValueId>();
        for (int nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
            CompiledNode node = nodes.get(nodeIndex);
            for (int inputIndex = 0; inputIndex < node.inputs().size(); inputIndex++) {
                ValueId input = node.inputs().get(inputIndex);
                if (!valuesById.containsKey(input)) {
                    throw new IllegalArgumentException(
                            "nodes["
                                    + nodeIndex
                                    + "].inputs["
                                    + inputIndex
                                    + "] is absent from values: "
                                    + input);
                }
                projectedInputIds.add(input);
            }
            for (int outputIndex = 0; outputIndex < node.outputs().size(); outputIndex++) {
                ValueId output = node.outputs().get(outputIndex);
                if (!valuesById.containsKey(output)) {
                    throw new IllegalArgumentException(
                            "nodes["
                                    + nodeIndex
                                    + "].outputs["
                                    + outputIndex
                                    + "] is absent from values: "
                                    + output);
                }
            }
        }

        for (int valueIndex = 0; valueIndex < values.size(); valueIndex++) {
            GraphValue value = values.get(valueIndex);
            LogicalMemoryRequirement requirement = requirementsById.get(value.id());
            if (requirement == null) {
                throw new IllegalArgumentException(
                        "memoryRequirements has no entry for values["
                                + valueIndex
                                + "].id "
                                + value.id());
            }
            if (!requirement.descriptor().equals(value.descriptor())) {
                throw new IllegalArgumentException(
                        "memoryRequirements["
                                + requirementIndexes.get(value.id())
                                + "].descriptor does not match values["
                                + valueIndex
                                + "].descriptor for "
                                + value.id());
            }
        }
        for (int requirementIndex = 0;
                requirementIndex < memoryRequirements.size();
                requirementIndex++) {
            LogicalMemoryRequirement requirement = memoryRequirements.get(requirementIndex);
            if (!valuesById.containsKey(requirement.valueId())) {
                throw new IllegalArgumentException(
                        "memoryRequirements["
                                + requirementIndex
                                + "].valueId is absent from values: "
                                + requirement.valueId());
            }
        }

        var copiedConstants = new LinkedHashMap<ValueId, ScalarValue>();
        for (Map.Entry<ValueId, ScalarValue> entry : constants.entrySet()) {
            ValueId valueId = Objects.requireNonNull(entry.getKey(), "constants key");
            ScalarValue constant =
                    Objects.requireNonNull(entry.getValue(), "constants[" + valueId + "]");
            GraphValue value = valuesById.get(valueId);
            if (value == null) {
                throw new IllegalArgumentException(
                        "constants key is absent from values: " + valueId);
            }
            LogicalMemoryRequirement requirement = requirementsById.get(valueId);
            if (!projectedInputIds.contains(valueId)
                    || requirement.producerPartition().isPresent()) {
                throw new IllegalArgumentException(
                        "constants key is not a projected graph input: " + valueId);
            }
            if (constant.dataType() != value.descriptor().dataType()) {
                throw new IllegalArgumentException(
                        "constants["
                                + valueId
                                + "] data type "
                                + constant.dataType()
                                + " does not match descriptor data type "
                                + value.descriptor().dataType());
            }
            copiedConstants.put(valueId, constant);
        }
        constants = Collections.unmodifiableMap(copiedConstants);
    }

    /**
     * Returns the analyzed partition recipe.
     *
     * @return the exact non-null immutable partition reference supplied at construction
     */
    @Override
    public PlannedPartition partition() {
        return partition;
    }

    /**
     * Returns the validated nodes in exact partition order.
     *
     * @return non-null immutable ordered snapshot containing the exact supplied node references
     */
    @Override
    public List<CompiledNode> nodes() {
        return nodes;
    }

    /**
     * Returns the projected logical graph values in supplied deterministic order.
     *
     * @return non-null immutable ordered snapshot of unique, fully static graph values
     */
    @Override
    public List<GraphValue> values() {
        return values;
    }

    /**
     * Returns the projected logical-memory requirements in supplied deterministic order.
     *
     * @return non-null immutable ordered snapshot with exactly one matching entry per value
     */
    @Override
    public List<LogicalMemoryRequirement> memoryRequirements() {
        return memoryRequirements;
    }

    /**
     * Returns projected exact-typed compile-time logical splats.
     *
     * @return non-null immutable map in supplied encounter order; keys are projected graph inputs
     *     and values are the exact supplied immutable scalar references
     */
    @Override
    public Map<ValueId, ScalarValue> constants() {
        return constants;
    }

    /**
     * Returns the backend's opaque immutable analysis input.
     *
     * @return exact non-null backend-owned input reference supplied at construction
     */
    @Override
    public I backendInputs() {
        return backendInputs;
    }
}
