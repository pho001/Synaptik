package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.ProducerlessPublishedValueResource;
import io.github.pho001.synaptik.prepare.PreparedScheduleContribution;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.prepare.PreparedWorkspaceAssignment;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.BooleanSupplier;

/**
 * Assembles the complete deterministic Runtime recipe for one non-empty maximal CPU partition.
 *
 * <p>The immutable assembler creates a representation-creation prefix, one exact finalized CPU
 * executable occurrence, and a forward-then-gradient publication suffix in stable Prepare-supplied
 * publication order. Source-only published callers retain borrowed representations and constants
 * use initialized-buffer recipes, without gaining an executable occurrence. It constructs recipes
 * only: no creator is invoked and no physical
 * resource is allocated, initialized, borrowed, executed, or published during assembly.</p>
 */
public final class CpuPreparedScheduleAssembler
        implements PreparedScheduleAssembler, PreparedScheduleContributor {
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
     * Derives CPU physical geometry for one authenticated producerless caller or constant.
     *
     * <p>The contribution retains the exact value and logical-requirement references. A present
     * scalar identifies a static, resolved, non-gradient canonical constant; an empty scalar
     * identifies a caller with either an exact resolved static affine layout or the existing
     * unresolved sole-named-batch template. Prepare authenticates those source roles before this
     * callback. Geometry reads no payload and creates no representation, executable or resource.
     * Later schedule assembly retains caller borrowing and fresh per-run constant initialization.</p>
     *
     * @param value exact non-null stable graph value with supported CPU source geometry
     * @param logicalRequirement exact non-null matching producerless logical requirement
     * @param scalar non-null optional authenticated constant scalar, matching the descriptor's
     *     type when present; empty only for an authenticated caller
     * @return a new non-null immutable contribution retaining the checked fixed or batch-bound size
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalStateException if the owning CPU integration is closed
     * @throws IllegalArgumentException if source geometry or constant policy is unsupported
     * @throws ArithmeticException if referenced-span byte arithmetic or CPU geometry limits overflow
     */
    @Override
    public ProducerlessPublishedValueResource producerlessPublishedValue(
            GraphValue value,
            LogicalMemoryRequirement logicalRequirement,
            Optional<ScalarValue> scalar) {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(logicalRequirement, "logicalRequirement");
        Objects.requireNonNull(scalar, "scalar");
        if (!ownerOpen.getAsBoolean()) {
            throw new IllegalStateException("CPU backend integration is closed");
        }
        var descriptor = value.descriptor();
        if (scalar.isPresent()) {
            requireCanonicalConstant(value.id(), descriptor, scalar.orElseThrow());
        }
        return new ProducerlessPublishedValueResource(
                value,
                logicalRequirement,
                CpuCapabilityProvider.CPU_BACKEND_ID,
                CpuPartitionLowering.descriptorByteSize(descriptor),
                descriptor.dataType().byteWidth());
    }

    /**
     * Applies the source-only constant policy without changing consumed-constant initialization.
     *
     * @param requirement exact complete-graph logical requirement identifying the source role
     * @param scalar exact authenticated constant scalar
     * @throws IllegalArgumentException if a producerless consumerless published constant violates
     *     CPU's static, nongradient, resolved canonical descriptor or scalar-type policy
     */
    private static void requireSourceConstantPolicy(
            LogicalMemoryRequirement requirement, ScalarValue scalar) {
        if (requirement.producerPartition().isEmpty()
                && requirement.consumerPartitions().isEmpty() && requirement.graphOutput()) {
            requireCanonicalConstant(
                    requirement.valueId(), requirement.descriptor(), scalar);
        }
    }

    /**
     * Checks CPU's canonical producerless constant contract shared by automatic and explicit sources.
     *
     * @param valueId stable source identifier for diagnostics
     * @param descriptor exact authenticated constant descriptor
     * @param scalar exact authenticated constant scalar
     * @throws IllegalArgumentException if descriptor geometry, gradient eligibility or type disagrees
     */
    private static void requireCanonicalConstant(
            ValueId valueId, TensorDescriptor descriptor, ScalarValue scalar) {
        if (scalar.dataType() != descriptor.dataType()) {
            throw new IllegalArgumentException(
                    "CPU source-only published constant scalar type disagrees with descriptor: "
                            + valueId);
        }
        if (!descriptor.shape().isFullyStatic()) {
            throw new IllegalArgumentException(
                    "CPU source-only published constant has dynamic shape: " + valueId);
        }
        if (descriptor.requiresGrad()) {
            throw new IllegalArgumentException(
                    "CPU source-only published constant requires gradients: " + valueId);
        }
        if (descriptor.layout().isEmpty()) {
            throw new IllegalArgumentException(
                    "CPU source-only published constant has unresolved layout: " + valueId);
        }
        if (!descriptor.layout().orElseThrow()
                .equals(LayoutDescriptor.contiguous(descriptor.shape()))) {
            throw new IllegalArgumentException(
                    "CPU source-only published constant has non-canonical layout: " + valueId);
        }
    }


    /** {@inheritDoc} */
    @Override
    public io.github.pho001.synaptik.backend.contract.BackendId backendId() {
        return CpuCapabilityProvider.CPU_BACKEND_ID;
    }

    /**
     * Contributes every externally assigned CPU buffer representation and workspace creator.
     * Source-only constants, including explicit resource contributions, retain the callback's
     * canonical descriptor policy; constants consumed by ordinary partitions keep their existing
     * initializer behavior.
     *
     * @param context exact non-null complete shared schedule context
     * @return non-null immutable sparse CPU contribution
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalStateException if the owning integration is closed
     * @throws IllegalArgumentException if a CPU-assigned buffer lacks a graph descriptor or a
     *     source-only constant violates CPU's canonical source policy
     */
    @Override
    public PreparedScheduleContribution contribute(PreparedScheduleContext context) {
        Objects.requireNonNull(context, "context");
        if (!ownerOpen.getAsBoolean()) {
            throw new IllegalStateException("CPU backend integration is closed");
        }
        var requirements = context.logicalMemoryRequirements().stream().collect(
                java.util.stream.Collectors.toMap(
                        LogicalMemoryRequirement::valueId, requirement -> requirement));
        var bindable = new HashSet<>(context.bindableInputValueIds());
        var buffers = new ArrayList<PreparedScheduleContribution.Buffer>();
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            int representationIndex =
                    assignment.representationOwners().indexOf(CpuCapabilityProvider.CPU_BACKEND_ID);
            if (representationIndex < 0) {
                continue;
            }
            var requirement = requirements.get(assignment.valueId());
            if (requirement == null) {
                throw new IllegalArgumentException(
                        "prepared CPU buffer has no graph descriptor");
            }
            var descriptor = requirement.descriptor();
            PreparedRepresentationPlan.BufferPreparation preparation;
            if (representationIndex == 0 && bindable.contains(assignment.valueId())) {
                preparation = new PreparedRepresentationPlan.CallerInput();
            } else if (representationIndex == 0
                    && context.constants().containsKey(assignment.valueId())) {
                requireSourceConstantPolicy(
                        requirement, context.constants().get(assignment.valueId()));
                preparation = new PreparedRepresentationPlan.InitializedBuffer(
                        CpuRepresentationRecipes.initializedBuffer(
                                descriptor.dataType(),
                                context.memoryPlan().buffers().get(assignment.planIndex()),
                                context.constants().get(assignment.valueId())));
            } else {
                preparation = new PreparedRepresentationPlan.CreatedBuffer(
                        CpuRepresentationRecipes.buffer(
                                descriptor.dataType(),
                                context.memoryPlan().buffers().get(assignment.planIndex())));
            }
            buffers.add(new PreparedScheduleContribution.Buffer(
                    assignment, representationIndex, preparation));
        }

        var workspaces = new ArrayList<PreparedScheduleContribution.Workspace>();
        for (PreparedWorkspaceAssignment assignment : context.workspaceAssignments()) {
            if (!assignment.partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
                continue;
            }
            workspaces.add(new PreparedScheduleContribution.Workspace(
                    assignment,
                    CpuRepresentationRecipes.workspace(
                            context.memoryPlan().workspaces().get(assignment.planIndex()))));
        }
        return new PreparedScheduleContribution(
                CpuCapabilityProvider.CPU_BACKEND_ID, buffers, workspaces);
    }
    /**
     * Assembles one immutable CPU representation, execution, and publication schedule.
     * Source-only constants are checked against the same CPU canonical policy even when their
     * physical resources were supplied explicitly rather than discovered through the callback.
     *
     * @param context the non-null validated context containing exactly one non-empty compiled and
     *     prepared CPU partition, complete dense buffer assignments, and one exact memory plan
     * @return a non-null immutable schedule retaining the context's exact memory plan, finalized
     *     executable, and buffer-creation recipes; initialized recipes allocate and initialize
     *     one fresh representation only when each new run state is created
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalStateException if the outer CPU integration owner is closed
     * @throws IllegalArgumentException if the context does not contain exactly one non-empty
     *     CPU-owned compile/prepared partition, a prepared buffer has no graph descriptor, or a
     *     published value has no prepared buffer assignment, or a source-only constant violates
     *     CPU's canonical source policy
     */
    @Override
    public PreparedSchedule assemble(PreparedScheduleContext context) {
        Objects.requireNonNull(context, "context");
        if (!ownerOpen.getAsBoolean()) {
            throw new IllegalStateException("CPU backend integration is closed");
        }
        validateSoleCpuPartition(context);

        var requirements = context.logicalMemoryRequirements().stream().collect(
                java.util.stream.Collectors.toMap(
                        LogicalMemoryRequirement::valueId, requirement -> requirement));
        var bindable = new HashSet<>(context.bindableInputValueIds());
        var constants = context.constants();

        var bufferPreparations = new ArrayList<
                List<PreparedRepresentationPlan.BufferPreparation>>(
                context.memoryPlan().buffers().size());
        for (int index = 0; index < context.bufferAssignments().size(); index++) {
            PreparedBufferAssignment assignment = context.bufferAssignments().get(index);
            var requirement = requirements.get(assignment.valueId());
            if (requirement == null) {
                throw new IllegalArgumentException("prepared CPU buffer has no graph descriptor");
            }
            var descriptor = requirement.descriptor();
            if (bindable.contains(assignment.valueId())) {
                bufferPreparations.add(List.of(new PreparedRepresentationPlan.CallerInput()));
            } else if (constants.containsKey(assignment.valueId())) {
                requireSourceConstantPolicy(requirement, constants.get(assignment.valueId()));
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
        publications.addAll(context.publicationValueIds());
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
        if (context.plannedPartitions().size() != 1 || context.partitions().size() != 1) {
            throw new IllegalArgumentException(
                    "CPU schedule requires exactly one non-empty CPU partition");
        }
        var compiled = context.plannedPartitions().getFirst();
        var prepared = context.partitions().getFirst().partition();
        if (compiled.nodeIds().isEmpty() || prepared.nodeIds().isEmpty()
                || !compiled.owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)
                || !prepared.owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException(
                    "CPU schedule requires exactly one non-empty CPU partition");
        }
    }
}
