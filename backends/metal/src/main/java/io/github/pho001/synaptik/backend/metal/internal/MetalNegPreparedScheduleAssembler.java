package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedPartition;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
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
import java.util.Objects;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Assembles the exact sole-partition all-Metal NEG Runtime recipe.
 *
 * <p>Feed buffers borrow caller inputs in stable feed order, target buffers are freshly allocated
 * per run, one execution step invokes the whole partition, and publication steps retain the
 * stable Prepare publication order including repeated aliases. Mixed-owner composition is
 * rejected.</p>
 */
final class MetalNegPreparedScheduleAssembler implements PreparedScheduleAssembler {
    private final MetalDeviceContext context;
    private final MetalNegPreparationPlan plan;
    private final List<ValueId> publicationValueIds;

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
                preparation = List.of(new PreparedRepresentationPlan.CreatedBuffer(
                        () -> context.createBuffer(bytes)));
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

    private static PreparedBufferAssignment requireAssignment(
            Map<ValueId, PreparedBufferAssignment> assignments, ValueId valueId) {
        PreparedBufferAssignment assignment = assignments.get(valueId);
        if (assignment == null) throw new IllegalArgumentException(
                "Metal NEG boundary value has no prepared assignment: " + valueId);
        return assignment;
    }
}
