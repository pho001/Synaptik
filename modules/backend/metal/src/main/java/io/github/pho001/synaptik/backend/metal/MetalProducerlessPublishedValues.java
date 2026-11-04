package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.prepare.ProducerlessPublishedValueResource;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import java.util.Objects;
import java.util.Optional;

/**
 * Supplies storage-only Metal geometry and ordinary per-run recipes for authenticated source values.
 *
 * <p>Prepare remains the source-role and complete-graph authentication owner. This contribution
 * allocates no physical resource during preparation, creates no partition or executable, borrows
 * callers through the existing input path, and reuses the existing raw-carrier constant initializer.
 * The seven qualified non-FP8 storage types are supported; FP8 fails before resource declaration.</p>
 */
final class MetalProducerlessPublishedValues {
    private MetalProducerlessPublishedValues() {}

    /**
     * Declares the exact resolved physical source span without reading payload or acquiring storage.
     *
     * @param context exact non-null borrowed device context, required to be open
     * @param value exact authenticated non-null graph input
     * @param requirement exact non-null producerless, consumerless published requirement
     * @param scalar non-null authenticated scalar for constants, empty for callers
     * @return immutable geometry retaining the exact value and requirement references
     * @throws NullPointerException if an argument is null
     * @throws IllegalStateException if the context is closed
     * @throws IllegalArgumentException if source geometry or constant policy is unsupported
     * @throws ArithmeticException if the referenced byte span overflows
     */
    static ProducerlessPublishedValueResource geometry(
            MetalDeviceContext context,
            GraphValue value,
            LogicalMemoryRequirement requirement,
            Optional<ScalarValue> scalar) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(requirement, "requirement");
        Objects.requireNonNull(scalar, "scalar");
        if (context.isClosed()) {
            throw new IllegalStateException("Metal device context is closed");
        }
        TensorDescriptor descriptor = value.descriptor();
        scalar.ifPresent(constant -> requireConstant(descriptor, constant));
        return new ProducerlessPublishedValueResource(
                value,
                requirement,
                MetalCapabilityProvider.METAL_BACKEND_ID,
                new PreparedMemoryPlan.FixedByteSize(byteSize(descriptor)),
                descriptor.dataType().byteWidth());
    }

    /**
     * Computes exact physical bytes for a resolved static source, including offsets and empty views.
     *
     * @param descriptor exact non-null fully static descriptor with resolved Model layout
     * @return non-negative physical referenced span in bytes; allocator padding is excluded
     * @throws NullPointerException if the descriptor is null
     * @throws IllegalArgumentException if shape or layout is unresolved, or the carrier is FP8
     * @throws ArithmeticException if span times carrier width overflows
     */
    static long byteSize(TensorDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        MetalBackendRuntime.requireStorageType(descriptor.dataType());
        if (!descriptor.shape().isFullyStatic() || descriptor.layout().isEmpty()) {
            throw new IllegalArgumentException(
                    "Metal source publication requires a static resolved descriptor");
        }
        return Math.multiplyExact(
                descriptor.layout().orElseThrow().referencedElementSpan(),
                descriptor.dataType().byteWidth());
    }

    /**
     * Supplies a source-only recipe before ordinary partition-specific contribution checks.
     *
     * @param context exact non-null borrowed device context
     * @param schedule exact non-null shared authenticated schedule facts
     * @param assignment exact non-null assignment from the schedule
     * @return the ordinary caller or initialized-buffer recipe, empty for partition-owned values
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if source role, descriptor or assigned geometry disagrees
     * @throws IllegalStateException if context closure has begun
     */
    static Optional<PreparedRepresentationPlan.BufferPreparation> contribution(
            MetalDeviceContext context,
            PreparedScheduleContext schedule,
            PreparedBufferAssignment assignment) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(schedule, "schedule");
        Objects.requireNonNull(assignment, "assignment");
        LogicalMemoryRequirement requirement = schedule.logicalMemoryRequirements().stream()
                .filter(candidate -> candidate.valueId().equals(assignment.valueId()))
                .findFirst().orElseThrow(() -> new IllegalArgumentException(
                        "Metal assignment has no logical requirement"));
        if (requirement.producerPartition().isPresent()
                || !requirement.consumerPartitions().isEmpty()
                || !requirement.graphOutput()) {
            return Optional.empty();
        }
        if (assignment.representationOwners().size() != 1
                || !assignment.representationOwners().getFirst()
                        .equals(MetalCapabilityProvider.METAL_BACKEND_ID)) {
            throw new IllegalArgumentException("Metal source requires one exact Metal owner");
        }
        GraphValue value = schedule.graphValues().stream()
                .filter(candidate -> candidate.id().equals(assignment.valueId()))
                .findFirst().orElseThrow();
        ScalarValue scalar = schedule.constants().get(assignment.valueId());
        boolean caller = schedule.bindableInputValueIds().contains(assignment.valueId());
        if (caller == (scalar != null)) {
            throw new IllegalArgumentException("Metal source must be exactly one caller or constant");
        }
        ProducerlessPublishedValueResource resource = geometry(
                context, value, requirement, Optional.ofNullable(scalar));
        long bytes = resource.size().fixedByteSize().orElseThrow();
        PreparedMemoryPlan.BufferEntry entry = schedule.memoryPlan().buffers()
                .get(assignment.planIndex());
        if (entry.byteSize() != bytes || entry.byteAlignment() != resource.byteAlignment()) {
            throw new IllegalArgumentException("Metal source assignment geometry disagrees");
        }
        if (caller) {
            return Optional.of(new PreparedRepresentationPlan.CallerInput());
        }
        return Optional.of(new PreparedRepresentationPlan.InitializedBuffer(
                (requestedBytes, alignment) -> {
                    if (requestedBytes != bytes || alignment != resource.byteAlignment()) {
                        throw new IllegalArgumentException("Metal source initializer geometry disagrees");
                    }
                    return MetalPreparedSplatResource.createRunBuffer(context, bytes, scalar);
                }));
    }

    /**
     * Checks the existing source-only canonical, static, nongradient constant policy.
     *
     * @param descriptor exact non-null authenticated source descriptor
     * @param scalar exact non-null authenticated constant scalar
     * @throws IllegalArgumentException if carrier, gradient eligibility or geometry disagrees
     */
    private static void requireConstant(TensorDescriptor descriptor, ScalarValue scalar) {
        if (scalar.dataType() != descriptor.dataType()
                || descriptor.requiresGrad()
                || !descriptor.shape().isFullyStatic()
                || descriptor.layout().isEmpty()
                || !descriptor.layout().orElseThrow()
                        .equals(LayoutDescriptor.contiguous(descriptor.shape()))) {
            throw new IllegalArgumentException(
                    "Metal source constant requires matching type and static nongradient canonical geometry");
        }
    }
}
