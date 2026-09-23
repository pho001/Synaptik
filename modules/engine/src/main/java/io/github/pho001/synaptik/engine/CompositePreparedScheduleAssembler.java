package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.prepare.PreparedScheduleContribution;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.prepare.PreparedWorkspaceAssignment;
import io.github.pho001.synaptik.prepare.ProducerlessPublishedConstantResource;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Assembles one complete CPU/Metal schedule from ordered owner-local physical contributions.
 *
 * <p>The assembler is created for one prepare transaction with direct registered adapter
 * references in registration order. It fills each shared Prepare-assigned representation and
 * workspace coordinate exactly once, emits the single representation-creation prefix, then walks
 * prepared partitions in compile order. Before each consumer executable it inserts at most one
 * transfer from representation zero to that consumer owner's position for each logical value.
 * Publications retain representation zero and form the required dense suffix.</p>
 */
final class CompositePreparedScheduleAssembler implements PreparedScheduleAssembler {
    private final Map<BackendId, EngineBackendComposition> adapters;

    /**
     * Captures exact adapters for one ordinary preparation.
     *
     * @param adapters non-null non-empty owner-keyed adapters in deterministic registration order
     */
    CompositePreparedScheduleAssembler(Map<BackendId, EngineBackendComposition> adapters) {
        Objects.requireNonNull(adapters, "adapters");
        if (adapters.isEmpty()) {
            throw new IllegalArgumentException("adapters must not be empty");
        }
        var copy = new LinkedHashMap<BackendId, EngineBackendComposition>();
        adapters.forEach((owner, adapter) -> copy.put(
                Objects.requireNonNull(owner, "adapter owner"),
                Objects.requireNonNull(adapter, "adapter")));
        this.adapters = Collections.unmodifiableMap(copy);
    }

    @Override
    public ProducerlessPublishedConstantResource producerlessPublishedConstant(
            GraphValue value,
            LogicalMemoryRequirement logicalRequirement,
            ScalarValue scalar) {
        for (EngineBackendComposition adapter : adapters.values()) {
            if (adapter instanceof CpuEngineBackendComposition) {
                return adapter.scheduleAssembler().producerlessPublishedConstant(
                        value, logicalRequirement, scalar);
            }
        }
        throw new IllegalArgumentException(
                "producerless published constants require a registered CPU adapter");
    }

    @Override
    public PreparedSchedule assemble(PreparedScheduleContext context) {
        Objects.requireNonNull(context, "context");
        validateOwnerOrder(context);

        var bufferPreparations = emptyBufferPreparations(context);
        PreparedRepresentationPlan.WorkspaceCreator[] workspaceCreators =
                new PreparedRepresentationPlan.WorkspaceCreator[
                        context.memoryPlan().workspaces().size()];
        Set<BackendId> usedOwners = usedOwners(context);
        for (Map.Entry<BackendId, EngineBackendComposition> entry : adapters.entrySet()) {
            if (!usedOwners.contains(entry.getKey())) {
                continue;
            }
            PreparedScheduleContributor contributor = entry.getValue().scheduleContributor();
            if (!contributor.backendId().equals(entry.getKey())) {
                throw new IllegalArgumentException(
                        "schedule contributor backend identity disagrees with registered owner");
            }
            PreparedScheduleContribution contribution = contributor.contribute(context);
            if (!contribution.owner().equals(entry.getKey())) {
                throw new IllegalArgumentException(
                        "schedule contribution owner disagrees with registered owner");
            }
            applyContribution(
                    context, contribution, bufferPreparations, workspaceCreators);
        }
        requireCompleteContributions(context, bufferPreparations, workspaceCreators);

        List<List<PreparedRepresentationPlan.BufferPreparation>> immutableBuffers =
                bufferPreparations.stream().map(List::copyOf).toList();
        var representationPlan = new PreparedRepresentationPlan(
                context.memoryPlan(), immutableBuffers, List.of(workspaceCreators));
        var steps = new ArrayList<PreparedSchedule.Step>();
        steps.add(new PreparedSchedule.RepresentationCreationStep(representationPlan));

        Map<ValueId, PreparedBufferAssignment> assignments = assignmentMap(context);
        Set<TransferCoordinate> emittedTransfers = new HashSet<>();
        for (int partitionIndex = 0;
                partitionIndex < context.plannedPartitions().size();
                partitionIndex++) {
            PlannedPartition partition = context.plannedPartitions().get(partitionIndex);
            for (LogicalMemoryRequirement requirement : context.logicalMemoryRequirements()) {
                if (!containsExact(requirement.consumerPartitions(), partition)) {
                    continue;
                }
                PreparedBufferAssignment assignment = assignments.get(requirement.valueId());
                if (assignment == null) {
                    // Partition-local virtual values have logical requirements but no shared slot.
                    continue;
                }
                int destinationIndex = assignment.representationOwners().indexOf(partition.owner());
                if (destinationIndex < 0) {
                    throw new IllegalArgumentException(
                            "consumer owner has no prepared representation position");
                }
                if (destinationIndex == 0
                        || !emittedTransfers.add(new TransferCoordinate(
                                assignment.planIndex(), destinationIndex))) {
                    continue;
                }
                BackendId sourceOwner = assignment.representationOwners().getFirst();
                EngineBackendComposition source = requireAdapter(sourceOwner);
                EngineBackendComposition destination = requireAdapter(partition.owner());
                steps.add(new PreparedSchedule.BufferTransferStep(source.prepareTransferTo(
                        destination,
                        context.memoryPlan(),
                        assignment.planIndex(),
                        0,
                        destinationIndex,
                        requirement.descriptor())));
            }
            steps.add(new PreparedSchedule.ExecutionStep(
                    context.partitions().get(partitionIndex).executable()));
        }

        for (int resultIndex = 0;
                resultIndex < context.publicationValueIds().size();
                resultIndex++) {
            ValueId valueId = context.publicationValueIds().get(resultIndex);
            PreparedBufferAssignment assignment = assignments.get(valueId);
            if (assignment == null) {
                throw new IllegalArgumentException(
                        "published value has no prepared buffer assignment: " + valueId);
            }
            steps.add(new PreparedSchedule.PublicationStep(new PreparedPublication(
                    context.memoryPlan(), assignment.planIndex(), 0, resultIndex)));
        }
        return new PreparedSchedule(context.memoryPlan(), steps);
    }

    private EngineBackendComposition requireAdapter(BackendId owner) {
        EngineBackendComposition adapter = adapters.get(owner);
        if (adapter == null) {
            throw new IllegalArgumentException("no registered adapter for backend owner " + owner);
        }
        return adapter;
    }

    private static ArrayList<ArrayList<PreparedRepresentationPlan.BufferPreparation>>
            emptyBufferPreparations(PreparedScheduleContext context) {
        var result = new ArrayList<
                ArrayList<PreparedRepresentationPlan.BufferPreparation>>(
                context.bufferAssignments().size());
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            var representations = new ArrayList<PreparedRepresentationPlan.BufferPreparation>(
                    assignment.representationOwners().size());
            for (int index = 0; index < assignment.representationOwners().size(); index++) {
                representations.add(null);
            }
            result.add(representations);
        }
        return result;
    }

    private static Set<BackendId> usedOwners(PreparedScheduleContext context) {
        var owners = new HashSet<BackendId>();
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            owners.addAll(assignment.representationOwners());
        }
        for (PreparedWorkspaceAssignment assignment : context.workspaceAssignments()) {
            owners.add(assignment.partition().owner());
        }
        return owners;
    }

    private static void applyContribution(
            PreparedScheduleContext context,
            PreparedScheduleContribution contribution,
            ArrayList<ArrayList<PreparedRepresentationPlan.BufferPreparation>> bufferPreparations,
            PreparedRepresentationPlan.WorkspaceCreator[] workspaceCreators) {
        for (PreparedScheduleContribution.Buffer buffer : contribution.buffers()) {
            int bufferIndex = buffer.assignment().planIndex();
            if (bufferIndex < 0
                    || bufferIndex >= context.bufferAssignments().size()
                    || context.bufferAssignments().get(bufferIndex) != buffer.assignment()) {
                throw new IllegalArgumentException(
                        "buffer contribution does not retain the assigned context entry");
            }
            var representations = bufferPreparations.get(bufferIndex);
            int representationIndex = buffer.representationIndex();
            if (representationIndex >= representations.size()) {
                throw new IllegalArgumentException(
                        "buffer contribution representation position is out of range");
            }
            if (representations.set(representationIndex, buffer.preparation()) != null) {
                throw new IllegalArgumentException(
                        "multiple contributions target one buffer representation position");
            }
        }
        for (PreparedScheduleContribution.Workspace workspace : contribution.workspaces()) {
            int workspaceIndex = workspace.assignment().planIndex();
            if (workspaceIndex < 0
                    || workspaceIndex >= context.workspaceAssignments().size()
                    || context.workspaceAssignments().get(workspaceIndex) != workspace.assignment()) {
                throw new IllegalArgumentException(
                        "workspace contribution does not retain the assigned context entry");
            }
            if (workspaceCreators[workspaceIndex] != null) {
                throw new IllegalArgumentException(
                        "multiple contributions target one workspace position");
            }
            workspaceCreators[workspaceIndex] = workspace.creator();
        }
    }

    private static void requireCompleteContributions(
            PreparedScheduleContext context,
            ArrayList<ArrayList<PreparedRepresentationPlan.BufferPreparation>> bufferPreparations,
            PreparedRepresentationPlan.WorkspaceCreator[] workspaceCreators) {
        for (int bufferIndex = 0; bufferIndex < bufferPreparations.size(); bufferIndex++) {
            for (int representationIndex = 0;
                    representationIndex < bufferPreparations.get(bufferIndex).size();
                    representationIndex++) {
                if (bufferPreparations.get(bufferIndex).get(representationIndex) == null) {
                    throw new IllegalArgumentException(
                            "no physical contribution for buffer "
                                    + bufferIndex
                                    + " representation "
                                    + representationIndex);
                }
            }
        }
        for (int workspaceIndex = 0; workspaceIndex < workspaceCreators.length; workspaceIndex++) {
            if (workspaceCreators[workspaceIndex] == null) {
                throw new IllegalArgumentException(
                        "no physical contribution for workspace " + workspaceIndex);
            }
        }
        if (context.bufferAssignments().size() != bufferPreparations.size()) {
            throw new AssertionError("buffer contribution cardinality changed");
        }
    }

    private static Map<ValueId, PreparedBufferAssignment> assignmentMap(
            PreparedScheduleContext context) {
        var assignments = new HashMap<ValueId, PreparedBufferAssignment>();
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            assignments.put(assignment.valueId(), assignment);
        }
        return assignments;
    }

    private static void validateOwnerOrder(PreparedScheduleContext context) {
        Map<ValueId, LogicalMemoryRequirement> requirements = new HashMap<>();
        for (LogicalMemoryRequirement requirement : context.logicalMemoryRequirements()) {
            requirements.put(requirement.valueId(), requirement);
        }
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            LogicalMemoryRequirement requirement = requirements.get(assignment.valueId());
            if (requirement == null) {
                throw new IllegalArgumentException(
                        "prepared buffer assignment has no logical memory requirement");
            }
            List<BackendId> expected = ownerOrder(requirement);
            if (!expected.isEmpty() && !assignment.representationOwners().equals(expected)) {
                throw new IllegalArgumentException(
                        "prepared representation-owner order disagrees with logical ownership");
            }
        }
    }

    /** Returns producer-first, then first-encounter distinct consumer owner order. */
    static List<BackendId> ownerOrder(LogicalMemoryRequirement requirement) {
        Objects.requireNonNull(requirement, "requirement");
        var owners = new ArrayList<BackendId>();
        requirement.producerPartition().ifPresent(partition -> owners.add(partition.owner()));
        for (PlannedPartition consumer : requirement.consumerPartitions()) {
            if (!owners.contains(consumer.owner())) {
                owners.add(consumer.owner());
            }
        }
        return List.copyOf(owners);
    }

    private static boolean containsExact(
            List<PlannedPartition> partitions, PlannedPartition candidate) {
        for (PlannedPartition partition : partitions) {
            if (partition == candidate) {
                return true;
            }
        }
        return false;
    }

    private record TransferCoordinate(int bufferIndex, int destinationRepresentationIndex) {}
}
