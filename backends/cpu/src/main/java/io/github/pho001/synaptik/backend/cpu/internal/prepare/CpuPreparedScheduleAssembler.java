package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.function.BooleanSupplier;

/**
 * Assembles the complete deterministic Runtime recipe for one non-empty maximal CPU partition.
 *
 * <p>The immutable assembler creates a representation-creation prefix, one exact finalized CPU
 * executable occurrence, and a forward-then-gradient publication suffix in Compiler publication
 * order. It constructs recipes only: no creator is invoked and no physical resource is allocated,
 * initialized, borrowed, executed, or published during assembly.</p>
 */
public final class CpuPreparedScheduleAssembler implements PreparedScheduleAssembler {
    private final BooleanSupplier ownerOpen;

    /**
     * Creates an immutable assembler guarded by its outer lifecycle owner.
     *
     * @param ownerOpen non-null thread-safe query that returns {@code true} only while assembly is
     *     allowed; the assembler retains the query without taking lifecycle ownership
     * @throws NullPointerException if {@code ownerOpen} is {@code null}
     */
    public CpuPreparedScheduleAssembler(BooleanSupplier ownerOpen) {
        this.ownerOpen = Objects.requireNonNull(ownerOpen, "ownerOpen");
    }

    /**
     * Assembles one immutable CPU representation, execution, and publication schedule.
     *
     * @param context the non-null validated context containing exactly one non-empty compiled and
     *     prepared CPU partition, complete dense buffer assignments, and one exact memory plan
     * @return a non-null immutable schedule retaining the context's exact memory plan and
     *     finalized executable
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalStateException if the outer CPU integration owner is closed
     * @throws IllegalArgumentException if the context does not contain exactly one non-empty
     *     CPU-owned compile/prepared partition, a prepared buffer has no graph descriptor, or a
     *     published value has no prepared buffer assignment
     */
    @Override
    public PreparedSchedule assemble(PreparedScheduleContext context) {
        Objects.requireNonNull(context, "context");
        if (!ownerOpen.getAsBoolean()) {
            throw new IllegalStateException("CPU backend integration is closed");
        }
        validateSoleCpuPartition(context);

        var descriptors = context.artifacts().graph().values().stream().collect(
                java.util.stream.Collectors.toMap(value -> value.id(), value -> value.descriptor()));
        var bindable = new HashSet<>(context.artifacts().constants().bindableInputs());
        var constants = new HashMap<ValueId,
                io.github.pho001.synaptik.model.datatype.ScalarValue>();
        context.artifacts().constants().constantSources()
                .forEach(source -> constants.put(source.valueId(), source.value()));

        var bufferPreparations = new ArrayList<
                List<PreparedRepresentationPlan.BufferPreparation>>(
                context.memoryPlan().buffers().size());
        for (int index = 0; index < context.bufferAssignments().size(); index++) {
            PreparedBufferAssignment assignment = context.bufferAssignments().get(index);
            var descriptor = descriptors.get(assignment.valueId());
            if (descriptor == null) {
                throw new IllegalArgumentException("prepared CPU buffer has no graph descriptor");
            }
            if (bindable.contains(assignment.valueId())) {
                bufferPreparations.add(List.of(new PreparedRepresentationPlan.CallerInput()));
            } else if (constants.containsKey(assignment.valueId())) {
                bufferPreparations.add(List.of(new PreparedRepresentationPlan.InitializedBuffer(
                        CpuRepresentationRecipes.initializedBuffer(
                                descriptor.dataType(), context.memoryPlan().buffers().get(index),
                                constants.get(assignment.valueId())))));
            } else {
                bufferPreparations.add(List.of(new PreparedRepresentationPlan.CreatedBuffer(
                        CpuRepresentationRecipes.buffer(
                                descriptor.dataType(), context.memoryPlan().buffers().get(index)))));
            }
        }

        var workspaceCreators = context.memoryPlan().workspaces().stream()
                .map(CpuRepresentationRecipes::workspace)
                .toList();
        var representationPlan = new PreparedRepresentationPlan(context.memoryPlan(),
                bufferPreparations, workspaceCreators);

        var steps = new ArrayList<PreparedSchedule.Step>();
        steps.add(new PreparedSchedule.RepresentationCreationStep(representationPlan));
        steps.add(new PreparedSchedule.ExecutionStep(
                context.partitions().getFirst().executable()));

        var assignments = new HashMap<ValueId, PreparedBufferAssignment>();
        context.bufferAssignments().forEach(assignment ->
                assignments.put(assignment.valueId(), assignment));
        var publications = new ArrayList<ValueId>();
        context.artifacts().publication().forwardBindings()
                .forEach(binding -> publications.add(binding.valueId()));
        context.artifacts().publication().gradientBindings()
                .forEach(binding -> publications.add(binding.valueId()));
        for (int resultIndex = 0; resultIndex < publications.size(); resultIndex++) {
            PreparedBufferAssignment assignment = assignments.get(publications.get(resultIndex));
            if (assignment == null) {
                throw new IllegalArgumentException(
                        "published CPU value has no prepared buffer assignment");
            }
            steps.add(new PreparedSchedule.PublicationStep(new PreparedPublication(
                    context.memoryPlan(), assignment.planIndex(), 0, resultIndex)));
        }
        return new PreparedSchedule(context.memoryPlan(), steps);
    }

    /**
     * Enforces this assembler's complete-schedule ownership boundary before recipe construction.
     *
     * @param context the non-null validated shared schedule context
     * @throws IllegalArgumentException if compiled and prepared coverage is not exactly one
     *     non-empty CPU-owned partition
     */
    private static void validateSoleCpuPartition(PreparedScheduleContext context) {
        if (context.artifacts().partitions().size() != 1 || context.partitions().size() != 1) {
            throw new IllegalArgumentException(
                    "CPU schedule requires exactly one non-empty CPU partition");
        }
        var compiled = context.artifacts().partitions().getFirst();
        var prepared = context.partitions().getFirst().partition();
        if (compiled.nodeIds().isEmpty() || prepared.nodeIds().isEmpty()
                || !compiled.owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)
                || !prepared.owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException(
                    "CPU schedule requires exactly one non-empty CPU partition");
        }
    }
}
