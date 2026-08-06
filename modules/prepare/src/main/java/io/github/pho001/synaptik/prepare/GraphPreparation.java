package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.InitializedBuffer;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.BufferTransferStep;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.ExecutionStep;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.PublicationStep;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.RepresentationCreationStep;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Performs complete shared graph preparation from immutable compile artifacts to one Runtime
 * execution recipe.
 *
 * <p>The operation projects every backend context before analysis, invokes analysis and
 * finalization in compile-partition order, exposes complete immutable facts to one explicit
 * schedule assembler, validates the returned recipe, and returns the exact prepared memory plan
 * and schedule in one {@link PreparedExecution}. It performs no physical work, execution,
 * backend discovery, tuning, or dynamic binding.</p>
 */
public final class GraphPreparation {
    private GraphPreparation() {}

    /**
     * Prepares one complete immutable compiled graph with explicit positional collaborators.
     *
     * @param artifacts exact non-null compile artifacts to prepare
     * @param preparations non-null positional preparation list, exactly one per compile
     *     partition; membership is snapshotted before backend work
     * @param scheduleAssembler exact non-null synchronous immutable recipe assembler, invoked
     *     once only after every finalizer succeeds and never retained
     * @return one non-null immutable reusable prepared execution retaining the exact assigned
     *     memory plan and validated schedule
     * @throws NullPointerException if a top-level input, preparation element, backend analysis,
     *     or assembled schedule is null
     * @throws IllegalArgumentException if positional coverage, projected analysis identity,
     *     finalized associations, or schedule structure is inconsistent
     */
    public static PreparedExecution prepare(
            CompileArtifacts artifacts,
            List<? extends PartitionPreparation<?, ?>> preparations,
            PreparedScheduleAssembler scheduleAssembler) {
        Objects.requireNonNull(artifacts, "artifacts");
        Objects.requireNonNull(preparations, "preparations");
        Objects.requireNonNull(scheduleAssembler, "scheduleAssembler");

        for (int index = 0; index < preparations.size(); index++) {
            Objects.requireNonNull(preparations.get(index), "preparations[" + index + "]");
        }
        preparations = List.copyOf(preparations);
        if (preparations.size() != artifacts.partitions().size()) {
            throw new IllegalArgumentException(
                    "preparations size must equal compile partition count "
                            + artifacts.partitions().size());
        }

        Projection projection = new Projection(artifacts);
        var invocations = new ArrayList<AnalysisInvocation>(preparations.size());
        for (int index = 0; index < preparations.size(); index++) {
            invocations.add(createInvocation(
                    index,
                    artifacts.partitions().get(index),
                    preparations.get(index),
                    projection));
        }

        var entries = new ArrayList<BackendPartitionFinalizationHandoff.Entry<?, ?>>(
                invocations.size());
        for (AnalysisInvocation invocation : invocations) {
            entries.add(invocation.analyze());
        }
        validateCrossPartitionDeclarations(artifacts.memory().requirements(), entries);

        BackendPartitionFinalizationHandoff.Result handoff =
                BackendPartitionFinalizationHandoff.finalizePartitions(
                        artifacts.partitions(), entries);
        PreparedScheduleContext context = new PreparedScheduleContext(
                artifacts,
                handoff.memoryPlan(),
                handoff.partitions(),
                handoff.bufferAssignments());
        PreparedSchedule schedule = Objects.requireNonNull(
                scheduleAssembler.assemble(context), "scheduleAssembler returned null");
        validateSchedule(context, schedule);
        return new PreparedExecution(handoff.memoryPlan(), schedule);
    }

    /**
     * Requires every partition at a logical cross-partition boundary to declare that value.
     * Values confined to one partition remain optional so backend-private fusion can keep them
     * virtual. This check interprets only Planning producer/consumer facts and shared declarations.
     */
    private static void validateCrossPartitionDeclarations(
            List<LogicalMemoryRequirement> requirements,
            List<? extends BackendPartitionFinalizationHandoff.Entry<?, ?>> entries) {
        var declarations = new HashMap<PlannedPartition, Set<ValueId>>();
        for (var entry : entries) {
            var values = declarations.computeIfAbsent(entry.analysis().partition(),
                    ignored -> new HashSet<>());
            entry.analysis().requirements().forEach(requirement -> {
                if (requirement instanceof io.github.pho001.synaptik.prepare.analysis
                        .PreparationResourceRequirement.Buffer buffer) {
                    values.add(buffer.valueId());
                }
            });
        }
        for (LogicalMemoryRequirement requirement : requirements) {
            var boundaryPartitions = new java.util.LinkedHashSet<PlannedPartition>();
            requirement.producerPartition().ifPresent(boundaryPartitions::add);
            boundaryPartitions.addAll(requirement.consumerPartitions());
            if (boundaryPartitions.size() < 2) continue;
            for (PlannedPartition partition : boundaryPartitions) {
                if (!declarations.getOrDefault(partition, Set.of()).contains(requirement.valueId())) {
                    throw new IllegalArgumentException(
                            "cross-partition value " + requirement.valueId()
                                    + " must be declared by partition " + partition);
                }
            }
        }
    }

    private static <I extends BackendAnalysisInputs, P extends BackendPreparationPlan>
            AnalysisInvocation createInvocation(
                    int index,
                    PlannedPartition partition,
                    PartitionPreparation<I, P> preparation,
                    Projection projection) {
        PrepareContext<I> context = projection.context(partition, preparation.backendInputs());
        return () -> {
            BackendPartitionAnalysis<P> analysis = Objects.requireNonNull(
                    preparation.preparer().analyze(context),
                    "preparations[" + index + "].preparer returned null");
            if (analysis.partition() != partition) {
                throw new IllegalArgumentException(
                        "preparations[" + index
                                + "].analysis partition does not match compile partition");
            }
            return new BackendPartitionFinalizationHandoff.Entry<>(
                    context, analysis, preparation.finalizer());
        };
    }

    private static void validateSchedule(
            PreparedScheduleContext context, PreparedSchedule schedule) {
        PreparedMemoryPlan memoryPlan = context.memoryPlan();
        if (schedule.memoryPlan() != memoryPlan) {
            throw new IllegalArgumentException(
                    "schedule memory plan does not match prepared memory plan");
        }

        PreparedRepresentationPlan representationPlan = null;
        if (!schedule.steps().isEmpty()
                && schedule.steps().getFirst() instanceof RepresentationCreationStep creation) {
            representationPlan = creation.representationPlan();
        }
        if ((!memoryPlan.buffers().isEmpty() || !memoryPlan.workspaces().isEmpty())
                && representationPlan == null) {
            throw new IllegalArgumentException(
                    "non-empty prepared memory plan requires a representation creation occurrence");
        }

        Map<ValueId, PreparedBufferAssignment> assignments = assignmentIndex(context);
        validateSourceRepresentations(context, representationPlan, assignments);
        validateExecutions(context, schedule);
        validateRepresentationCoordinates(schedule, representationPlan);
        validatePublications(context, schedule, representationPlan, assignments);
    }

    private static Map<ValueId, PreparedBufferAssignment> assignmentIndex(
            PreparedScheduleContext context) {
        var assignments = new HashMap<ValueId, PreparedBufferAssignment>();
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            assignments.put(assignment.valueId(), assignment);
        }
        return assignments;
    }

    private static void validateSourceRepresentations(
            PreparedScheduleContext context,
            PreparedRepresentationPlan representationPlan,
            Map<ValueId, PreparedBufferAssignment> assignments) {
        List<ValueId> bindableInputs = context.artifacts().constants().bindableInputs();
        for (ValueId valueId : bindableInputs) {
            requireAssignment(assignments, valueId);
        }
        for (CompileConstantPlan.ConstantSource source
                : context.artifacts().constants().constantSources()) {
            requireAssignment(assignments, source.valueId());
        }

        for (CompileConstantPlan.ConstantSource source
                : context.artifacts().constants().constantSources()) {
            ValueId valueId = source.valueId();
            List<PreparedRepresentationPlan.BufferPreparation> preparations =
                    representationPlan.bufferPreparations()
                            .get(assignments.get(valueId).planIndex());
            for (PreparedRepresentationPlan.BufferPreparation preparation : preparations) {
                if (preparation instanceof CallerInput) {
                    throw new IllegalArgumentException(
                            "constant source " + valueId
                                    + " must not use a caller-input representation");
                }
            }
        }

        var callerBufferIndexes = new ArrayList<Integer>();
        if (representationPlan != null) {
            for (int bufferIndex = 0;
                    bufferIndex < representationPlan.bufferPreparations().size();
                    bufferIndex++) {
                for (PreparedRepresentationPlan.BufferPreparation preparation
                        : representationPlan.bufferPreparations().get(bufferIndex)) {
                    if (preparation instanceof CallerInput) {
                        callerBufferIndexes.add(bufferIndex);
                    }
                }
            }
        }
        if (callerBufferIndexes.size() != bindableInputs.size()) {
            throw new IllegalArgumentException(
                    "caller-input occurrence count must equal bindable input count "
                            + bindableInputs.size());
        }
        for (int index = 0; index < bindableInputs.size(); index++) {
            ValueId valueId = bindableInputs.get(index);
            if (callerBufferIndexes.get(index) != assignments.get(valueId).planIndex()) {
                throw new IllegalArgumentException(
                        "caller-input occurrence " + index + " does not match bindableInputs["
                                + index + "] " + valueId);
            }
        }

        Set<ValueId> constants = new HashSet<>();
        for (CompileConstantPlan.ConstantSource source
                : context.artifacts().constants().constantSources()) {
            ValueId valueId = source.valueId();
            constants.add(valueId);
            List<PreparedRepresentationPlan.BufferPreparation> preparations =
                    representationPlan.bufferPreparations()
                            .get(assignments.get(valueId).planIndex());
            boolean initialized = false;
            for (PreparedRepresentationPlan.BufferPreparation preparation : preparations) {
                initialized |= preparation instanceof InitializedBuffer;
            }
            if (!initialized) {
                throw new IllegalArgumentException(
                        "constant source " + valueId
                                + " requires an initialized buffer representation");
            }
        }
        if (representationPlan != null) {
            for (int bufferIndex = 0;
                    bufferIndex < representationPlan.bufferPreparations().size();
                    bufferIndex++) {
                ValueId valueId = context.bufferAssignments().get(bufferIndex).valueId();
                for (PreparedRepresentationPlan.BufferPreparation preparation
                        : representationPlan.bufferPreparations().get(bufferIndex)) {
                    if (preparation instanceof InitializedBuffer && !constants.contains(valueId)) {
                        throw new IllegalArgumentException(
                                "buffer " + valueId
                                        + " has an initialized representation but is not a constant source");
                    }
                }
            }
        }
    }

    private static void validateExecutions(
            PreparedScheduleContext context, PreparedSchedule schedule) {
        var executions = new ArrayList<PreparedExecutable>();
        for (PreparedSchedule.Step step : schedule.steps()) {
            if (step instanceof ExecutionStep execution) {
                executions.add(execution.executable());
            }
        }
        if (executions.size() != context.partitions().size()) {
            throw new IllegalArgumentException(
                    "execution occurrence count must equal prepared partition count "
                            + context.partitions().size());
        }
        for (int index = 0; index < executions.size(); index++) {
            if (executions.get(index) != context.partitions().get(index).executable()) {
                throw new IllegalArgumentException(
                        "execution occurrence " + index
                                + " does not match preparedPartitions[" + index + "].executable");
            }
        }
    }

    private static void validateRepresentationCoordinates(
            PreparedSchedule schedule, PreparedRepresentationPlan representationPlan) {
        for (int stepIndex = 0; stepIndex < schedule.steps().size(); stepIndex++) {
            PreparedSchedule.Step step = schedule.steps().get(stepIndex);
            if (step instanceof ExecutionStep execution) {
                for (int selectionIndex = 0;
                        selectionIndex < execution.executable().bufferSelectionCount();
                        selectionIndex++) {
                    PreparedExecutable.BufferSelection selection =
                            execution.executable().bufferSelection(selectionIndex);
                    validateRepresentationIndex(
                            representationPlan,
                            stepIndex,
                            "executable.bufferSelections[" + selectionIndex + "]",
                            selection.bufferIndex(),
                            selection.representationIndex());
                }
            } else if (step instanceof BufferTransferStep transferStep) {
                PreparedBufferTransfer transfer = transferStep.transfer();
                validateRepresentationIndex(
                        representationPlan,
                        stepIndex,
                        "transfer.source",
                        transfer.bufferIndex(),
                        transfer.sourceRepresentationIndex());
                validateRepresentationIndex(
                        representationPlan,
                        stepIndex,
                        "transfer.destination",
                        transfer.bufferIndex(),
                        transfer.destinationRepresentationIndex());
            } else if (step instanceof PublicationStep publicationStep) {
                PreparedPublication publication = publicationStep.publication();
                validateRepresentationIndex(
                        representationPlan,
                        stepIndex,
                        "publication",
                        publication.bufferIndex(),
                        publication.representationIndex());
            }
        }
    }

    private static void validateRepresentationIndex(
            PreparedRepresentationPlan representationPlan,
            int stepIndex,
            String role,
            int bufferIndex,
            int representationIndex) {
        int count = representationPlan == null
                ? 0
                : representationPlan.bufferPreparations().get(bufferIndex).size();
        if (representationIndex >= count) {
            throw new IllegalArgumentException(
                    "steps[" + stepIndex + "] " + role
                            + " representationIndex out of creation-plan range: "
                            + representationIndex);
        }
    }

    private static void validatePublications(
            PreparedScheduleContext context,
            PreparedSchedule schedule,
            PreparedRepresentationPlan representationPlan,
            Map<ValueId, PreparedBufferAssignment> assignments) {
        var expected = new ArrayList<ValueId>();
        context.artifacts().publication().forwardBindings()
                .forEach(binding -> expected.add(binding.valueId()));
        context.artifacts().publication().gradientBindings()
                .forEach(binding -> expected.add(binding.valueId()));

        var publications = new ArrayList<PreparedPublication>();
        for (PreparedSchedule.Step step : schedule.steps()) {
            if (step instanceof PublicationStep publication) {
                publications.add(publication.publication());
            }
        }
        if (publications.size() != expected.size()) {
            throw new IllegalArgumentException(
                    "publication occurrence count must equal requested result count "
                            + expected.size());
        }
        for (int index = 0; index < expected.size(); index++) {
            ValueId valueId = expected.get(index);
            PreparedBufferAssignment assignment = requireAssignment(assignments, valueId);
            if (publications.get(index).bufferIndex() != assignment.planIndex()) {
                throw new IllegalArgumentException(
                        "publication occurrence " + index
                                + " bufferIndex does not match requested value " + valueId);
            }
        }
    }

    private static PreparedBufferAssignment requireAssignment(
            Map<ValueId, PreparedBufferAssignment> assignments, ValueId valueId) {
        PreparedBufferAssignment assignment = assignments.get(valueId);
        if (assignment == null) {
            throw new IllegalArgumentException(
                    "requested value has no prepared buffer assignment: " + valueId);
        }
        return assignment;
    }

    private interface AnalysisInvocation {
        BackendPartitionFinalizationHandoff.Entry<?, ?> analyze();
    }

    private static final class Projection {
        private final CompileArtifacts artifacts;
        private final Map<NodeId, CompiledNode> nodes;
        private final Map<ValueId, LogicalMemoryRequirement> requirements;

        private Projection(CompileArtifacts artifacts) {
            this.artifacts = artifacts;
            nodes = new HashMap<>();
            for (CompiledNode node : artifacts.graph().nodes()) {
                nodes.put(node.id(), node);
            }
            requirements = new HashMap<>();
            for (LogicalMemoryRequirement requirement : artifacts.memory().requirements()) {
                requirements.put(requirement.valueId(), requirement);
            }
        }

        private <I extends BackendAnalysisInputs> PrepareContext<I> context(
                PlannedPartition partition, I backendInputs) {
            var partitionNodes = new ArrayList<CompiledNode>(partition.nodeIds().size());
            var projectedIds = new HashSet<ValueId>();
            for (NodeId nodeId : partition.nodeIds()) {
                CompiledNode node = nodes.get(nodeId);
                partitionNodes.add(node);
                projectedIds.addAll(node.inputs());
                projectedIds.addAll(node.outputs());
            }

            var values = new ArrayList<GraphValue>();
            var memoryRequirements = new ArrayList<LogicalMemoryRequirement>();
            for (GraphValue value : artifacts.graph().values()) {
                if (projectedIds.contains(value.id())) {
                    values.add(value);
                    memoryRequirements.add(requirements.get(value.id()));
                }
            }
            var constants = new LinkedHashMap<ValueId, ScalarValue>();
            for (CompileConstantPlan.ConstantSource source
                    : artifacts.constants().constantSources()) {
                if (projectedIds.contains(source.valueId())) {
                    constants.put(source.valueId(), source.value());
                }
            }
            return new PrepareContext<>(
                    partition,
                    partitionNodes,
                    values,
                    memoryRequirements,
                    constants,
                    backendInputs);
        }
    }
}
