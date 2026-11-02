package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.util.Objects;

/**
 * Contributes immutable physical geometry for one caller or constant graph input published
 * without a producer or consumer.
 *
 * <p>The exact Model value, Planning requirement and Runtime size recipe are retained by
 * identity. Complete graph preparation authenticates membership, source role, descriptor policy
 * and coverage before analysis. This record neither reads payload nor allocates, initializes,
 * acquires or transfers a representation. Caller storage remains borrowed; constant
 * initialization belongs to the existing per-run initializer.</p>
 *
 * @param value exact non-null immutable graph value to retain and return by identity from
 *     {@link #value()}
 * @param logicalRequirement exact non-null immutable producerless, consumerless graph-output
 *     requirement for {@code value}, retained and returned by identity from
 *     {@link #logicalRequirement()}
 * @param owner exact non-null backend representation owner
 * @param size exact non-null checked physical size recipe, retained unchanged, including
 *     structurally symbolic recipes whose evaluated size is always zero
 * @param byteAlignment positive power-of-two physical byte alignment supplied by concrete
 *     composition and returned unchanged from {@link #byteAlignment()}
 */
public record ProducerlessPublishedValueResource(
        GraphValue value,
        LogicalMemoryRequirement logicalRequirement,
        BackendId owner,
        PreparedMemoryPlan.ByteSize size,
        long byteAlignment) {
    /**
     * Validates the standalone association and generic Runtime-compatible physical geometry.
     *
     * Construction retains both immutable reference components by identity and has no side
     * effect beyond fail-fast validation. It does not prove graph membership, infer physical
     * geometry, allocate a buffer, initialize a representation, or transfer ownership.
     *
     * @param value exact non-null immutable graph value to retain by identity
     * @param logicalRequirement exact non-null immutable logical requirement to retain by
     *     identity
     * @param owner exact non-null representation owner
     * @param size exact non-null physical size recipe to retain without evaluation
     * @param byteAlignment positive power-of-two supplied physical byte alignment
     * @throws NullPointerException if a reference component is null, checked in component order
     * @throws IllegalArgumentException if the value and requirement disagree, the logical role
     *     is not producerless/consumerless/output-required, or alignment is invalid
     */
    public ProducerlessPublishedValueResource {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(logicalRequirement, "logicalRequirement");
        Objects.requireNonNull(owner, "owner");
        Objects.requireNonNull(size, "size");
        if (!logicalRequirement.valueId().equals(value.id())) {
            throw new IllegalArgumentException(
                    "logicalRequirement.valueId must match value.id");
        }
        if (!logicalRequirement.descriptor().equals(value.descriptor())) {
            throw new IllegalArgumentException(
                    "logicalRequirement.descriptor must match value.descriptor");
        }
        if (logicalRequirement.producerPartition().isPresent()) {
            throw new IllegalArgumentException("logicalRequirement must be producerless");
        }
        if (!logicalRequirement.consumerPartitions().isEmpty()) {
            throw new IllegalArgumentException("logicalRequirement must be consumerless");
        }
        if (!logicalRequirement.graphOutput()) {
            throw new IllegalArgumentException(
                    "logicalRequirement must require graph output");
        }
        if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
            throw new IllegalArgumentException(
                    "byteAlignment must be a positive power of two");
        }
    }
}
