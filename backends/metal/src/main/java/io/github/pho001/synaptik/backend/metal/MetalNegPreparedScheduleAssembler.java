package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedPartition;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContribution;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.prepare.PreparedWorkspaceAssignment;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Objects;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Supplies Metal physical contributions for supported elementwise and terminal-affine operations
 * and assembles the legacy sole-partition route.
 *
 * <p>Shared mixed-owner composition uses {@link #contribute(PreparedScheduleContext)} and owns
 * the global step order. This class remains responsible only for Metal physical creation,
 * authenticated dense affine publication targets, and the direct route-local execution step. The
 * legacy {@link #assemble(PreparedScheduleContext)} entry accepts only one Metal partition and
 * delegates to the same route-local assembly.</p>
 */
final class MetalNegPreparedScheduleAssembler
        implements PreparedScheduleAssembler, PreparedScheduleContributor {
    private final MetalDeviceContext context;
    private final MetalNegPreparationPlan plan;
    private final List<ValueId> publicationValueIds;

    /**
     * Creates the stable integration assembler that recovers the finalized immutable plan from
     * the exact Metal executable during cold schedule assembly.
     *
     * @param context non-null borrowed device context
     * @throws NullPointerException if {@code context} is {@code null}
     */
    MetalNegPreparedScheduleAssembler(MetalDeviceContext context) {
        this.context = Objects.requireNonNull(context, "context");
        this.plan = null;
        this.publicationValueIds = null;
    }

    /**
     * Creates one immutable assembler for an analyzed route and its exact device context.
     *
     * @param context non-null borrowed context used by run-owned output creators
     * @param plan non-null analyzed whole-partition lowering
     * @param publicationValueIds non-null result identities in exact Prepare publication order,
     *     including repeated aliases
     * @throws NullPointerException if a reference or publication element is {@code null}
     */
    MetalNegPreparedScheduleAssembler(
            MetalDeviceContext context,
            MetalNegPreparationPlan plan,
            List<ValueId> publicationValueIds) {
        this.context = Objects.requireNonNull(context, "context");
        this.plan = Objects.requireNonNull(plan, "plan");
        if (plan.context() != context) {
            throw new IllegalArgumentException("Metal NEG analysis/schedule context mismatch");
        }
        this.publicationValueIds = List.copyOf(publicationValueIds);
    }

    /** {@inheritDoc} */
    @Override
    public io.github.pho001.synaptik.backend.contract.BackendId backendId() {
        return MetalCapabilityProvider.METAL_BACKEND_ID;
    }

    /**
     * Contributes every externally assigned Metal buffer representation and workspace creator.
     *
     * @param scheduleContext exact non-null complete shared schedule context
     * @return non-null immutable sparse Metal contribution
     * @throws NullPointerException if {@code scheduleContext} is null
     * @throws IllegalArgumentException if an assigned descriptor, constant, executable, or
     *     workspace is outside the current Metal NEG domain
     */
    @Override
    public PreparedScheduleContribution contribute(PreparedScheduleContext scheduleContext) {
        Objects.requireNonNull(scheduleContext, "scheduleContext");
        var descriptors = scheduleContext.graphValues().stream().collect(
                java.util.stream.Collectors.toMap(
                        value -> value.id(), value -> value.descriptor()));
        var bindable = new HashSet<>(scheduleContext.bindableInputValueIds());
        var buffers = new ArrayList<PreparedScheduleContribution.Buffer>();
        for (PreparedBufferAssignment assignment : scheduleContext.bufferAssignments()) {
            int representationIndex = assignment.representationOwners().indexOf(
                    MetalCapabilityProvider.METAL_BACKEND_ID);
            if (representationIndex < 0) {
                continue;
            }
            var descriptor = descriptors.get(assignment.valueId());
            long bytes = scheduleContext.memoryPlan().buffers()
                    .get(assignment.planIndex()).byteSize();
            Optional<MetalBufferRepresentation.DenseAffinePublication> affinePublication =
                    denseAffinePublication(
                            scheduleContext.partitions(),
                            assignment.valueId(),
                            descriptor,
                            bytes);
            if (descriptor == null
                    || descriptor.dataType()
                            != io.github.pho001.synaptik.model.datatype.DataType.FLOAT32
                    || !descriptor.shape().isFullyStatic()
                    || descriptor.layout().isEmpty()
                    || (!descriptor.layout().orElseThrow().equals(
                            io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(
                                    descriptor.shape()))
                            && affinePublication.isEmpty())) {
                throw new IllegalArgumentException(
                        "Metal assigned buffer requires canonical or authenticated affine FLOAT32");
            }
            PreparedRepresentationPlan.BufferPreparation preparation;
            if (affinePublication.isPresent()) {
                var publication = affinePublication.orElseThrow();
                preparation = new PreparedRepresentationPlan.CreatedBuffer(
                        () -> context.createBuffer(bytes, publication));
            } else if (representationIndex == 0 && bindable.contains(assignment.valueId())) {
                preparation = new PreparedRepresentationPlan.CallerInput();
            } else if (representationIndex == 0
                    && scheduleContext.constants().containsKey(assignment.valueId())) {
                var scalar = scheduleContext.constants().get(assignment.valueId());
                if (scalar.dataType()
                        != io.github.pho001.synaptik.model.datatype.DataType.FLOAT32) {
                    throw new IllegalArgumentException(
                            "Metal NEG initialized buffer requires FLOAT32 scalar");
                }
                int bits = Float.floatToRawIntBits(scalar.float32Value());
                preparation = new PreparedRepresentationPlan.InitializedBuffer(
                        () -> createSplatBuffer(bytes, bits));
            } else {
                preparation = new PreparedRepresentationPlan.CreatedBuffer(
                        () -> context.createBuffer(bytes));
            }
            buffers.add(new PreparedScheduleContribution.Buffer(
                    assignment, representationIndex, preparation));
        }

        var workspaces = new ArrayList<PreparedScheduleContribution.Workspace>();
        for (PreparedWorkspaceAssignment assignment : scheduleContext.workspaceAssignments()) {
            if (!assignment.partition().owner().equals(
                    MetalCapabilityProvider.METAL_BACKEND_ID)) {
                continue;
            }
            MetalNegPreparedExecutable executable = null;
            for (PreparedPartition partition : scheduleContext.partitions()) {
                if (partition.partition() == assignment.partition()
                        && partition.executable() instanceof MetalNegPreparedExecutable metal) {
                    executable = metal;
                    break;
                }
            }
            if (executable == null
                    || executable.preparationPlan().route()
                            != MetalNegPreparationPlan.Route.MPSGRAPH) {
                throw new IllegalArgumentException(
                        "Metal NEG workspace has no exact MPSGraph executable");
            }
            MetalNegPreparationPlan executablePlan = executable.preparationPlan();
            int pointerCount = Math.addExact(
                    executablePlan.feedValueIds().size(),
                    executablePlan.targetValueIds().size());
            workspaces.add(new PreparedScheduleContribution.Workspace(
                    assignment,
                    () -> new MetalNegPreparedExecutable.AddressWorkspace(
                            context, pointerCount)));
        }
        return new PreparedScheduleContribution(
                MetalCapabilityProvider.METAL_BACKEND_ID, buffers, workspaces);
    }

    /**
     * Constructs one representation prefix, one execution, and an ordered publication suffix.
     *
     * @param scheduleContext non-null complete finalized shared facts
     * @return non-null immutable complete Runtime schedule
     * @throws NullPointerException if {@code scheduleContext} is {@code null}
     * @throws IllegalArgumentException if the graph is not the exact sole all-Metal partition or
     *     assignments do not cover feeds and targets in the analyzed order
     */
    @Override
    public PreparedSchedule assemble(PreparedScheduleContext scheduleContext) {
        Objects.requireNonNull(scheduleContext, "scheduleContext");
        if (plan == null) {
            if (scheduleContext.partitions().size() != 1
                    || !(scheduleContext.partitions().getFirst().executable()
                            instanceof MetalNegPreparedExecutable executable)) {
                throw new IllegalArgumentException(
                        "Metal NEG schedule requires one finalized Metal executable");
            }
            return new MetalNegPreparedScheduleAssembler(
                    context, executable.preparationPlan(),
                    scheduleContext.publicationValueIds()).assemble(scheduleContext);
        }
        if (scheduleContext.plannedPartitions().size() != 1
                || scheduleContext.plannedPartitions().getFirst() != plan.partition()
                || scheduleContext.partitions().size() != 1
                || !scheduleContext.partitions().getFirst().partition().owner()
                        .equals(MetalCapabilityProvider.METAL_BACKEND_ID)) {
            throw new IllegalArgumentException(
                    "Metal NEG schedule requires one sole all-Metal partition");
        }
        return assembleRoute(scheduleContext.memoryPlan(),
                scheduleContext.partitions().getFirst(), scheduleContext.bufferAssignments());
    }

    /**
     * Assembles the already validated sole-partition Metal route from its typed shared facts.
     *
     * <p>This package-private entry point is the backend-local composition seam used when no
     * public Compiler artifact construction is available. It retains every identity and
     * validation rule of {@link #assemble(PreparedScheduleContext)}.</p>
     *
     * @param memoryPlan exact non-null finalized shared memory plan
     * @param preparedPartition exact non-null finalized Metal partition and executable
     * @param bufferAssignments non-null dense value assignments indexed by plan position
     * @return non-null immutable Runtime schedule
     * @throws NullPointerException if an argument or assignment is {@code null}
     * @throws IllegalArgumentException if identity, ownership, plan, assignment, or boundary
     *     coverage disagrees with the analyzed route
     */
    PreparedSchedule assembleRoute(
            PreparedMemoryPlan memoryPlan,
            PreparedPartition preparedPartition,
            List<PreparedBufferAssignment> bufferAssignments) {
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(preparedPartition, "preparedPartition");
        Objects.requireNonNull(bufferAssignments, "bufferAssignments");
        if (!preparedPartition.partition().owner().equals(MetalCapabilityProvider.METAL_BACKEND_ID)) {
            throw new IllegalArgumentException("Metal NEG schedule requires Metal ownership");
        }
        if (preparedPartition.partition() != plan.partition()) {
            throw new IllegalArgumentException(
                    "Metal NEG analyzed plan/scheduled partition identity mismatch");
        }
        if (!(preparedPartition.executable() instanceof MetalNegPreparedExecutable executable)
                || executable.preparationPlan() != plan) {
            throw new IllegalArgumentException(
                    "Metal analyzed plan/scheduled executable identity mismatch");
        }
        if (preparedPartition.executable().memoryPlan() != memoryPlan) {
            throw new IllegalArgumentException(
                    "Metal NEG scheduled executable/memory-plan identity mismatch");
        }
        if (bufferAssignments.size() != memoryPlan.buffers().size()) {
            throw new IllegalArgumentException(
                    "Metal NEG buffer assignments must cover the memory plan");
        }

        Map<ValueId, PreparedBufferAssignment> assignments = new HashMap<>();
        for (int index = 0; index < bufferAssignments.size(); index++) {
            PreparedBufferAssignment assignment = Objects.requireNonNull(
                    bufferAssignments.get(index), "bufferAssignments[" + index + "]");
            if (assignment.planIndex() != index
                    || assignment.slot() != memoryPlan.buffers().get(index).slot()
                    || assignments.putIfAbsent(assignment.valueId(), assignment) != null) {
                throw new IllegalArgumentException(
                        "Metal NEG buffer assignment order, slot, or value identity disagrees");
            }
        }
        var feeds = new HashSet<>(plan.feedValueIds());
        var targets = new HashSet<>(plan.targetValueIds());
        var preparations = new ArrayList<
                List<PreparedRepresentationPlan.BufferPreparation>>(Collections.nCopies(
                memoryPlan.buffers().size(), null));
        for (PreparedBufferAssignment assignment : bufferAssignments) {
            List<PreparedRepresentationPlan.BufferPreparation> preparation;
            if (feeds.contains(assignment.valueId())) {
                int feedIndex = plan.feedValueIds().indexOf(assignment.valueId());
                var splat = plan.feedSplats().get(feedIndex);
                if (splat.isPresent()) {
                    long bytes = plan.feedRequiredBytes()[feedIndex];
                    int bits = Float.floatToRawIntBits(splat.orElseThrow().float32Value());
                    preparation = List.of(new PreparedRepresentationPlan.InitializedBuffer(
                            () -> createSplatBuffer(bytes, bits)));
                } else {
                    preparation = List.of(new PreparedRepresentationPlan.CallerInput());
                }
            } else if (targets.contains(assignment.valueId())) {
                long bytes = memoryPlan.buffers()
                        .get(assignment.planIndex()).byteSize();
                Optional<MetalBufferRepresentation.DenseAffinePublication> publication =
                        executable.denseAffinePublication(assignment.valueId());
                preparation = List.of(new PreparedRepresentationPlan.CreatedBuffer(
                        () -> publication.isPresent()
                                ? context.createBuffer(bytes, publication.orElseThrow())
                                : context.createBuffer(bytes)));
            } else {
                throw new IllegalArgumentException(
                        "Metal NEG schedule contains an undeclared boundary buffer");
            }
            if (preparations.set(assignment.planIndex(), preparation) != null) {
                throw new IllegalArgumentException(
                        "Metal NEG schedule repeats a prepared buffer plan index");
            }
        }
        for (ValueId feed : plan.feedValueIds()) requireAssignment(assignments, feed);
        for (ValueId target : plan.targetValueIds()) requireAssignment(assignments, target);

        List<PreparedRepresentationPlan.WorkspaceCreator> workspaceCreators;
        if (plan.route() == MetalNegPreparationPlan.Route.MPSGRAPH) {
            int pointerCount = Math.addExact(
                    plan.feedValueIds().size(), plan.targetValueIds().size());
            workspaceCreators = List.of(() ->
                    new MetalNegPreparedExecutable.AddressWorkspace(context, pointerCount));
        } else {
            workspaceCreators = List.of();
        }
        var representationPlan = new PreparedRepresentationPlan(
                memoryPlan, preparations, workspaceCreators);
        var steps = new ArrayList<PreparedSchedule.Step>();
        steps.add(new PreparedSchedule.RepresentationCreationStep(representationPlan));
        steps.add(new PreparedSchedule.ExecutionStep(
                preparedPartition.executable()));

        for (int resultIndex = 0; resultIndex < publicationValueIds.size(); resultIndex++) {
            PreparedBufferAssignment assignment = requireAssignment(
                    assignments, publicationValueIds.get(resultIndex));
            if (!targets.contains(assignment.valueId())) {
                throw new IllegalArgumentException(
                        "published Metal NEG value is not an analyzed target");
            }
            steps.add(new PreparedSchedule.PublicationStep(new PreparedPublication(
                    memoryPlan, assignment.planIndex(), 0, resultIndex)));
        }
        return new PreparedSchedule(memoryPlan, steps);
    }

    private MetalBufferRepresentation createSplatBuffer(long bytes, int rawBits) {
        MetalBufferRepresentation buffer = context.createBuffer(bytes);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes, Integer.BYTES);
            long elements = bytes / Float.BYTES;
            for (long index = 0; index < elements; index++) {
                source.setAtIndex(JAVA_INT, index, rawBits);
            }
            buffer.upload(0L, source, 0L, bytes);
            return buffer;
        } catch (RuntimeException | Error failure) {
            try {
                buffer.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }


    private static Optional<MetalBufferRepresentation.DenseAffinePublication>
            denseAffinePublication(
                    List<PreparedPartition> partitions,
                    ValueId valueId,
                    TensorDescriptor descriptor,
                    long byteSize) {
        if (descriptor == null) {
            return Optional.empty();
        }
        for (PreparedPartition partition : partitions) {
            if (partition.executable() instanceof MetalNegPreparedExecutable executable) {
                MetalNegPreparationPlan candidate = executable.preparationPlan();
                if (candidate.authenticatesDenseAffineTarget(
                        valueId, descriptor, byteSize)) {
                    return executable.denseAffinePublication(valueId);
                }
            }
        }
        return Optional.empty();
    }
    private static PreparedBufferAssignment requireAssignment(
            Map<ValueId, PreparedBufferAssignment> assignments, ValueId valueId) {
        PreparedBufferAssignment assignment = assignments.get(valueId);
        if (assignment == null) throw new IllegalArgumentException(
                "Metal NEG boundary value has no prepared assignment: " + valueId);
        return assignment;
    }
}
