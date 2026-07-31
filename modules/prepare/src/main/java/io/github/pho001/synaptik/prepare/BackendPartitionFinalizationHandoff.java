package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Performs the package-internal complete-set assignment and backend-finalization handoff.
 *
 * <p>The operation validates the complete ordered partition set before deriving any assignment,
 * constructs every typed finalization before invoking a backend, and then invokes finalizers in
 * partition order. It creates immutable recipe associations only; it performs no physical
 * allocation, per-run binding, execution, scheduling, transfer, or publication.</p>
 */
final class BackendPartitionFinalizationHandoff {
    private BackendPartitionFinalizationHandoff() {}

    /**
     * Assigns one complete ordered analysis set and finalizes it against one shared memory plan.
     *
     * @param partitions non-null ordered expected partitions; elements must be non-null and
     *     unique by value
     * @param entries non-null ordered typed context, analysis, and finalizer associations; there
     *     must be exactly one non-null entry for each expected partition
     * @return a non-null immutable result retaining the shared plan and prepared partitions in
     *     expected partition order
     * @throws NullPointerException if a list, partition, entry, finalizer backend identity, or
     *     returned executable is null
     * @throws IllegalArgumentException if coverage, exact source identity, backend ownership,
     *     assignment geometry, or returned executable plan identity is inconsistent
     */
    static Result finalizePartitions(
            List<PlannedPartition> partitions,
            List<? extends Entry<?, ?>> entries) {
        Objects.requireNonNull(partitions, "partitions");
        Objects.requireNonNull(entries, "entries");

        var observedPartitions = new HashSet<PlannedPartition>();
        for (int index = 0; index < partitions.size(); index++) {
            PlannedPartition partition =
                    Objects.requireNonNull(partitions.get(index), "partitions[" + index + "]");
            if (!observedPartitions.add(partition)) {
                throw new IllegalArgumentException(
                        "partitions[" + index + "] duplicates " + partition);
            }
        }
        for (int index = 0; index < entries.size(); index++) {
            Objects.requireNonNull(entries.get(index), "entries[" + index + "]");
        }
        if (entries.size() != partitions.size()) {
            throw new IllegalArgumentException(
                    "entries size must equal partitions size " + partitions.size());
        }

        var firstBufferSources = new LinkedHashMap<ValueId, Source>();
        for (int index = 0; index < entries.size(); index++) {
            validateEntry(index, partitions.get(index), entries.get(index), firstBufferSources);
        }

        var buffers = new LinkedHashMap<ValueId, BufferAggregate>();
        var workspaceEntries = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        var assignmentLists = new ArrayList<List<PreparationResourceAssignment>>(entries.size());
        for (Entry<?, ?> entry : entries) {
            var assignments = new ArrayList<PreparationResourceAssignment>();
            for (PreparationResourceRequirement requirement : entry.analysis().requirements()) {
                switch (requirement) {
                    case PreparationResourceRequirement.Buffer buffer -> {
                        BufferAggregate aggregate = buffers.get(buffer.valueId());
                        if (aggregate == null) {
                            aggregate = new BufferAggregate(
                                    new BufferSlot(buffers.size()),
                                    buffers.size(),
                                    buffer.byteSize(),
                                    buffer.byteAlignment());
                            buffers.put(buffer.valueId(), aggregate);
                        } else {
                            aggregate.include(buffer);
                        }
                        assignments.add(new PreparationResourceAssignment.Buffer(
                                buffer, aggregate.slot, aggregate.planIndex));
                    }
                    case PreparationResourceRequirement.Workspace workspace -> {
                        int planIndex = workspaceEntries.size();
                        WorkspaceSlot slot = new WorkspaceSlot(planIndex);
                        workspaceEntries.add(new PreparedMemoryPlan.WorkspaceEntry(
                                slot, workspace.byteSize(), workspace.byteAlignment()));
                        assignments.add(new PreparationResourceAssignment.Workspace(
                                workspace, slot, planIndex));
                    }
                }
            }
            assignmentLists.add(List.copyOf(assignments));
        }

        var bufferEntries = new ArrayList<PreparedMemoryPlan.BufferEntry>(buffers.size());
        for (BufferAggregate aggregate : buffers.values()) {
            bufferEntries.add(new PreparedMemoryPlan.BufferEntry(
                    aggregate.slot, aggregate.byteSize, aggregate.byteAlignment));
        }
        PreparedMemoryPlan memoryPlan = new PreparedMemoryPlan(bufferEntries, workspaceEntries);

        var invocations = new ArrayList<FinalizationInvocation>(entries.size());
        for (int index = 0; index < entries.size(); index++) {
            invocations.add(createInvocation(entries.get(index), memoryPlan, assignmentLists.get(index)));
        }

        var preparedPartitions = new ArrayList<PreparedPartition>(entries.size());
        for (int index = 0; index < invocations.size(); index++) {
            PreparedExecutable executable = Objects.requireNonNull(
                    invocations.get(index).finalizePartition(),
                    "entries[" + index + "].finalizer returned null");
            if (executable.memoryPlan() != memoryPlan) {
                throw new IllegalArgumentException(
                        "entries[" + index
                                + "] executable memory plan does not match assigned memory plan");
            }
            preparedPartitions.add(new PreparedPartition(partitions.get(index), executable));
        }
        return new Result(memoryPlan, preparedPartitions);
    }

    private static void validateEntry(
            int entryIndex,
            PlannedPartition partition,
            Entry<?, ?> entry,
            Map<ValueId, Source> firstBufferSources) {
        if (entry.context().partition() != partition) {
            throw new IllegalArgumentException(
                    "entries[" + entryIndex
                            + "].context.partition does not match partitions[" + entryIndex + "]");
        }
        if (entry.analysis().partition() != entry.context().partition()) {
            throw new IllegalArgumentException(
                    "entries[" + entryIndex
                            + "].analysis.partition does not match context partition");
        }
        var backendId = Objects.requireNonNull(
                entry.finalizer().backendId(),
                "entries[" + entryIndex + "].finalizer.backendId");
        if (!backendId.equals(partition.owner())) {
            throw new IllegalArgumentException(
                    "entries[" + entryIndex + "].finalizer backendId " + backendId
                            + " does not match partition owner " + partition.owner());
        }

        var values = new LinkedHashMap<ValueId, GraphValue>();
        for (GraphValue value : entry.context().values()) {
            values.put(value.id(), value);
        }
        var logicalRequirements = new LinkedHashMap<ValueId, LogicalMemoryRequirement>();
        for (LogicalMemoryRequirement requirement : entry.context().memoryRequirements()) {
            logicalRequirements.put(requirement.valueId(), requirement);
        }
        for (int requirementIndex = 0;
                requirementIndex < entry.analysis().requirements().size();
                requirementIndex++) {
            PreparationResourceRequirement requirement =
                    entry.analysis().requirements().get(requirementIndex);
            if (requirement instanceof PreparationResourceRequirement.Buffer buffer) {
                GraphValue value = values.get(buffer.valueId());
                if (value == null) {
                    throw new IllegalArgumentException(
                            "entries[" + entryIndex + "].requirements[" + requirementIndex
                                    + "] buffer valueId is absent from context.values: "
                                    + buffer.valueId());
                }
                LogicalMemoryRequirement logicalRequirement =
                        logicalRequirements.get(buffer.valueId());
                Source first = firstBufferSources.get(buffer.valueId());
                if (first == null) {
                    firstBufferSources.put(
                            buffer.valueId(), new Source(value, logicalRequirement));
                } else {
                    if (first.value != value) {
                        throw new IllegalArgumentException(
                                "entries[" + entryIndex + "].requirements[" + requirementIndex
                                        + "] projected value reference does not match first declaration for "
                                        + buffer.valueId());
                    }
                    if (first.logicalRequirement != logicalRequirement) {
                        throw new IllegalArgumentException(
                                "entries[" + entryIndex + "].requirements[" + requirementIndex
                                        + "] logical requirement reference does not match first declaration for "
                                        + buffer.valueId());
                    }
                }
            }
        }
    }

    private static <I extends BackendAnalysisInputs, P extends BackendPreparationPlan>
            FinalizationInvocation createInvocation(
                    Entry<I, P> entry,
                    PreparedMemoryPlan memoryPlan,
                    List<PreparationResourceAssignment> assignments) {
        BackendPartitionFinalization<P> finalization =
                new BackendPartitionFinalization<>(entry.analysis(), memoryPlan, assignments);
        return () -> entry.finalizer().finalizePartition(finalization);
    }

    /**
     * Preserves one backend's typed analysis-to-finalizer association.
     *
     * @param <I> concrete backend-owned immutable analysis-input role
     * @param <P> corresponding concrete backend-owned immutable selected-plan role
     * @param context exact non-null validated context that produced the analysis
     * @param analysis exact non-null analysis to finalize
     * @param finalizer exact non-null owning-backend collaboration
     */
    record Entry<I extends BackendAnalysisInputs, P extends BackendPreparationPlan>(
            PrepareContext<I> context,
            BackendPartitionAnalysis<P> analysis,
            BackendPartitionFinalizer<P> finalizer) {
        /**
         * Retains one complete typed finalization entry.
         *
         * @param context exact non-null context to retain
         * @param analysis exact non-null analysis to retain
         * @param finalizer exact non-null finalizer to retain
         * @throws NullPointerException if any component is null
         */
        Entry {
            Objects.requireNonNull(context, "context");
            Objects.requireNonNull(analysis, "analysis");
            Objects.requireNonNull(finalizer, "finalizer");
        }
    }

    /**
     * Returns one shared plan and the immutable ordered prepared-partition associations.
     *
     * @param memoryPlan exact non-null shared prepared memory plan
     * @param partitions non-null prepared partitions in expected order; elements must be non-null
     */
    record Result(PreparedMemoryPlan memoryPlan, List<PreparedPartition> partitions) {
        /**
         * Validates and snapshots a complete handoff result.
         *
         * @param memoryPlan exact non-null plan to retain
         * @param partitions non-null ordered prepared partitions to snapshot
         * @throws NullPointerException if the plan, list, or an element is null
         */
        Result {
            Objects.requireNonNull(memoryPlan, "memoryPlan");
            Objects.requireNonNull(partitions, "partitions");
            for (int index = 0; index < partitions.size(); index++) {
                Objects.requireNonNull(partitions.get(index), "partitions[" + index + "]");
            }
            partitions = List.copyOf(partitions);
        }
    }

    private interface FinalizationInvocation {
        PreparedExecutable finalizePartition();
    }

    private static final class BufferAggregate {
        private final BufferSlot slot;
        private final int planIndex;
        private long byteSize;
        private long byteAlignment;

        private BufferAggregate(
                BufferSlot slot, int planIndex, long byteSize, long byteAlignment) {
            this.slot = slot;
            this.planIndex = planIndex;
            this.byteSize = byteSize;
            this.byteAlignment = byteAlignment;
        }

        private void include(PreparationResourceRequirement.Buffer requirement) {
            byteSize = Math.max(byteSize, requirement.byteSize());
            byteAlignment = Math.max(byteAlignment, requirement.byteAlignment());
        }
    }

    private static final class Source {
        private final GraphValue value;
        private final LogicalMemoryRequirement logicalRequirement;

        private Source(GraphValue value, LogicalMemoryRequirement logicalRequirement) {
            this.value = value;
            this.logicalRequirement = logicalRequirement;
        }
    }
}
