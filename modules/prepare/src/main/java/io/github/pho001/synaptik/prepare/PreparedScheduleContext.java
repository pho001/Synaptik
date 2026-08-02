package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Supplies complete immutable compile-to-prepared facts to one schedule assembler invocation.
 *
 * <p>The two lists are immutable membership snapshots retaining their exact elements. The value
 * proves positional partition and dense buffer-assignment associations but owns no executable
 * work, physical resource, or mutable run state.</p>
 *
 * @param artifacts exact non-null immutable compile artifacts
 * @param memoryPlan exact non-null shared prepared memory plan
 * @param partitions non-null prepared partitions in compile partition order
 * @param bufferAssignments non-null logical buffer associations in memory-plan buffer order
 */
public record PreparedScheduleContext(
        CompileArtifacts artifacts,
        PreparedMemoryPlan memoryPlan,
        List<PreparedPartition> partitions,
        List<PreparedBufferAssignment> bufferAssignments) {
    /**
     * Validates and snapshots one complete schedule-assembly context.
     *
     * @param artifacts exact non-null compile artifacts to retain
     * @param memoryPlan exact non-null prepared memory plan to retain
     * @param partitions non-null ordered prepared partitions to validate and snapshot
     * @param bufferAssignments non-null dense assignments to validate and snapshot
     * @throws NullPointerException if a component or indexed element is null
     * @throws IllegalArgumentException if partition coverage, exact reference associations,
     *     assignment order, slot identity, uniqueness, or graph membership is inconsistent
     */
    public PreparedScheduleContext {
        Objects.requireNonNull(artifacts, "artifacts");
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(partitions, "partitions");
        Objects.requireNonNull(bufferAssignments, "bufferAssignments");

        for (int index = 0; index < partitions.size(); index++) {
            Objects.requireNonNull(partitions.get(index), "partitions[" + index + "]");
        }
        for (int index = 0; index < bufferAssignments.size(); index++) {
            Objects.requireNonNull(
                    bufferAssignments.get(index), "bufferAssignments[" + index + "]");
        }
        partitions = List.copyOf(partitions);
        bufferAssignments = List.copyOf(bufferAssignments);

        if (partitions.size() != artifacts.partitions().size()) {
            throw new IllegalArgumentException(
                    "partitions size must equal compile partition count "
                            + artifacts.partitions().size());
        }
        for (int index = 0; index < partitions.size(); index++) {
            PreparedPartition partition = partitions.get(index);
            if (partition.partition() != artifacts.partitions().get(index)) {
                throw new IllegalArgumentException(
                        "partitions[" + index + "] does not retain artifacts.partitions[" + index + "]");
            }
            if (partition.executable().memoryPlan() != memoryPlan) {
                throw new IllegalArgumentException(
                        "partitions[" + index + "].executable does not retain memoryPlan");
            }
        }

        if (bufferAssignments.size() != memoryPlan.buffers().size()) {
            throw new IllegalArgumentException(
                    "bufferAssignments size must equal prepared buffer count "
                            + memoryPlan.buffers().size());
        }
        var graphValues = new HashSet<ValueId>();
        for (GraphValue value : artifacts.graph().values()) {
            graphValues.add(value.id());
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
            if (!graphValues.contains(assignment.valueId())) {
                throw new IllegalArgumentException(
                        "bufferAssignments[" + index + "].valueId is absent from artifacts.graph: "
                                + assignment.valueId());
            }
        }
    }
}
