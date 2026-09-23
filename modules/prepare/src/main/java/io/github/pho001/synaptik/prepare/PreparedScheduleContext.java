package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Supplies validated stable Model, Planning, Prepare, and Runtime facts to schedule assembly.
 *
 * <p>Every collection is an immutable membership snapshot retaining its exact immutable elements.
 * The value contains no Compiler aggregate, executable work, physical resource, or mutable run
 * state.</p>
 *
 * @param plannedPartitions non-null planned partitions in compile order
 * @param graphValues non-null graph values in stable graph order
 * @param logicalMemoryRequirements non-null requirements in stable graph-value order
 * @param bindableInputValueIds non-null caller-bindable source IDs in stable source order
 * @param constants non-null compile-time constants in stable source order
 * @param publicationValueIds non-null result IDs in exact publication order, including aliases
 * @param memoryPlan exact non-null shared prepared memory plan
 * @param partitions non-null prepared partitions in planned-partition order
 * @param bufferAssignments non-null logical buffer associations in memory-plan buffer order
 * @param workspaceAssignments non-null partition-owned workspace associations in memory-plan order
 */
public record PreparedScheduleContext(
        List<PlannedPartition> plannedPartitions,
        List<GraphValue> graphValues,
        List<LogicalMemoryRequirement> logicalMemoryRequirements,
        List<ValueId> bindableInputValueIds,
        Map<ValueId, ScalarValue> constants,
        List<ValueId> publicationValueIds,
        PreparedMemoryPlan memoryPlan,
        List<PreparedPartition> partitions,
        List<PreparedBufferAssignment> bufferAssignments,
        List<PreparedWorkspaceAssignment> workspaceAssignments) {
    /**
     * Validates and snapshots one complete stable schedule-assembly context.
     *
     * @param plannedPartitions non-null planned partitions in compile order; list structure is
     *     snapshotted and immutable elements are retained by identity
     * @param graphValues non-null stable graph values; list structure is snapshotted and
     *     immutable elements are retained by identity
     * @param logicalMemoryRequirements non-null graph-value-ordered logical requirements;
     *     structure is snapshotted and exact immutable elements are retained
     * @param bindableInputValueIds non-null caller-bindable source IDs in source order; list
     *     structure is snapshotted
     * @param constants non-null compile-time constant map; entries are copied in encounter order
     * @param publicationValueIds non-null result IDs in exact publication order, including
     *     repeated aliases; list structure is snapshotted
     * @param memoryPlan exact non-null shared prepared memory plan retained by identity
     * @param partitions non-null prepared partitions in planned-partition order; list structure
     *     is snapshotted and elements are retained by identity
     * @param bufferAssignments non-null logical assignments in memory-plan buffer order; list
     *     structure is snapshotted and elements are retained by identity
     * @param workspaceAssignments non-null plan-ordered workspace ownership assignments; list
     *     structure is snapshotted and exact immutable elements are retained
     * @throws NullPointerException if a component or indexed element is null
     * @throws IllegalArgumentException if partition, source, publication, assignment, slot,
     *     uniqueness, or graph membership is inconsistent
     */
    public PreparedScheduleContext {
        Objects.requireNonNull(plannedPartitions, "plannedPartitions");
        Objects.requireNonNull(graphValues, "graphValues");
        Objects.requireNonNull(logicalMemoryRequirements, "logicalMemoryRequirements");
        Objects.requireNonNull(bindableInputValueIds, "bindableInputValueIds");
        Objects.requireNonNull(constants, "constants");
        Objects.requireNonNull(publicationValueIds, "publicationValueIds");
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(partitions, "partitions");
        Objects.requireNonNull(bufferAssignments, "bufferAssignments");
        Objects.requireNonNull(workspaceAssignments, "workspaceAssignments");

        plannedPartitions = List.copyOf(plannedPartitions);
        graphValues = List.copyOf(graphValues);
        logicalMemoryRequirements = List.copyOf(logicalMemoryRequirements);
        bindableInputValueIds = List.copyOf(bindableInputValueIds);
        publicationValueIds = List.copyOf(publicationValueIds);
        var constantsCopy = new LinkedHashMap<ValueId, ScalarValue>();
        constants.forEach((key, value) -> constantsCopy.put(
                Objects.requireNonNull(key, "constants key"),
                Objects.requireNonNull(value, "constants[" + key + "]")));
        constants = Collections.unmodifiableMap(constantsCopy);
        partitions = List.copyOf(partitions);
        bufferAssignments = List.copyOf(bufferAssignments);
        workspaceAssignments = List.copyOf(workspaceAssignments);

        if (partitions.size() != plannedPartitions.size()) {
            throw new IllegalArgumentException(
                    "partitions size must equal planned partition count " + plannedPartitions.size());
        }
        for (int index = 0; index < partitions.size(); index++) {
            PreparedPartition partition = partitions.get(index);
            if (partition.partition() != plannedPartitions.get(index)) {
                throw new IllegalArgumentException(
                        "partitions[" + index + "] does not retain plannedPartitions[" + index + "]");
            }
            if (partition.executable().memoryPlan() != memoryPlan) {
                throw new IllegalArgumentException(
                        "partitions[" + index + "].executable does not retain memoryPlan");
            }
        }

        var graphValueIds = new HashSet<ValueId>();
        for (int index = 0; index < graphValues.size(); index++) {
            GraphValue value = graphValues.get(index);
            if (!graphValueIds.add(value.id())) {
                throw new IllegalArgumentException(
                        "graphValues[" + index + "].id duplicates " + value.id());
            }
        }
        if (logicalMemoryRequirements.size() != graphValues.size()) {
            throw new IllegalArgumentException(
                    "logicalMemoryRequirements size must equal graphValues size "
                            + graphValues.size());
        }
        for (int index = 0; index < logicalMemoryRequirements.size(); index++) {
            LogicalMemoryRequirement requirement = logicalMemoryRequirements.get(index);
            GraphValue value = graphValues.get(index);
            if (!requirement.valueId().equals(value.id())
                    || !requirement.descriptor().equals(value.descriptor())) {
                throw new IllegalArgumentException(
                        "logicalMemoryRequirements[" + index
                                + "] does not match graphValues[" + index + "]");
            }
        }
        for (ValueId valueId : bindableInputValueIds) requireGraphValue(
                graphValueIds, valueId, "bindable input");
        for (ValueId valueId : constants.keySet()) requireGraphValue(
                graphValueIds, valueId, "constant");
        for (ValueId valueId : publicationValueIds) requireGraphValue(
                graphValueIds, valueId, "publication");

        if (bufferAssignments.size() != memoryPlan.buffers().size()) {
            throw new IllegalArgumentException(
                    "bufferAssignments size must equal prepared buffer count "
                            + memoryPlan.buffers().size());
        }
        var assignedValues = new HashSet<ValueId>();
        for (int index = 0; index < bufferAssignments.size(); index++) {
            PreparedBufferAssignment assignment = bufferAssignments.get(index);
            if (assignment.planIndex() != index) {
                throw new IllegalArgumentException(
                        "bufferAssignments[" + index + "].planIndex must equal " + index);
            }
            if (assignment.slot() != memoryPlan.buffers().get(index).slot()) {
                throw new IllegalArgumentException(
                        "bufferAssignments[" + index + "].slot does not match memoryPlan.buffers["
                                + index + "]");
            }
            if (!assignedValues.add(assignment.valueId())) {
                throw new IllegalArgumentException(
                        "bufferAssignments[" + index + "].valueId duplicates "
                                + assignment.valueId());
            }
            requireGraphValue(graphValueIds, assignment.valueId(),
                    "bufferAssignments[" + index + "].valueId");
        }

        if (workspaceAssignments.size() != memoryPlan.workspaces().size()) {
            throw new IllegalArgumentException(
                    "workspaceAssignments size must equal prepared workspace count "
                            + memoryPlan.workspaces().size());
        }
        for (int index = 0; index < workspaceAssignments.size(); index++) {
            PreparedWorkspaceAssignment assignment = workspaceAssignments.get(index);
            if (assignment.planIndex() != index) {
                throw new IllegalArgumentException(
                        "workspaceAssignments[" + index + "].planIndex must equal " + index);
            }
            if (assignment.slot() != memoryPlan.workspaces().get(index).slot()) {
                throw new IllegalArgumentException(
                        "workspaceAssignments[" + index
                                + "].slot does not match memoryPlan.workspaces[" + index + "]");
            }
            boolean exactPartition = false;
            for (PlannedPartition partition : plannedPartitions) {
                exactPartition |= partition == assignment.partition();
            }
            if (!exactPartition) {
                throw new IllegalArgumentException(
                        "workspaceAssignments[" + index
                                + "].partition is absent from plannedPartitions");
            }
        }
    }

    private static void requireGraphValue(Set<ValueId> graphValues, ValueId valueId, String role) {
        Objects.requireNonNull(valueId, role);
        if (!graphValues.contains(valueId)) {
            throw new IllegalArgumentException(role + " is absent from graphValues: " + valueId);
        }
    }
}
