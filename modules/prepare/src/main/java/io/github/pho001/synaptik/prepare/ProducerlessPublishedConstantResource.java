package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import java.util.Objects;

/**
 * Contributes physical buffer geometry for one fully resolved compile-time constant that is
 * published without being produced or consumed by a graph partition.
 *
 * <p>The value retains the exact Compiler graph-value and Planning logical-requirement
 * references supplied by composition. Its component accessors return those same references and
 * the supplied primitive geometry without copying or mutation. Shared Prepare validates exact
 * reference membership and the complete source/publication role against one
 * {@code CompileArtifacts} instance before backend analysis. This record neither selects an
 * owner nor derives geometry, creates a representation, allocates storage, or materializes the
 * constant.</p>
 *
 * @param value exact non-null immutable graph value to retain and return by identity from
 *     {@link #value()}
 * @param logicalRequirement exact non-null immutable producerless, consumerless graph-output
 *     requirement for {@code value}, retained and returned by identity from
 *     {@link #logicalRequirement()}
 * @param byteSize non-negative physical buffer size in bytes, supplied by concrete composition
 *     and returned unchanged from {@link #byteSize()}
 * @param byteAlignment positive power-of-two physical byte alignment supplied by concrete
 *     composition and returned unchanged from {@link #byteAlignment()}
 */
public record ProducerlessPublishedConstantResource(
        GraphValue value,
        LogicalMemoryRequirement logicalRequirement,
        long byteSize,
        long byteAlignment) {
    /**
     * Validates the standalone association and generic Runtime-compatible physical geometry.
     *
     * Construction retains both immutable reference components by identity and has no side
     * effect beyond fail-fast validation. It does not prove artifact membership, infer physical
     * geometry, allocate a buffer, initialize a representation, or transfer ownership.
     *
     * @param value exact non-null immutable graph value to retain by identity
     * @param logicalRequirement exact non-null immutable logical requirement to retain by
     *     identity
     * @param byteSize non-negative supplied physical buffer size in bytes
     * @param byteAlignment positive power-of-two supplied physical byte alignment
     * @throws NullPointerException if {@code value} or {@code logicalRequirement} is null,
     *     checked in component order
     * @throws IllegalArgumentException if the references disagree by value identity or
     *     descriptor, the logical role is not producerless/consumerless/output-required, the
     *     descriptor is dynamic or unresolved, or the physical geometry is outside its required
     *     domain
     */
    public ProducerlessPublishedConstantResource {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(logicalRequirement, "logicalRequirement");
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
        if (!value.descriptor().shape().isFullyStatic()) {
            throw new IllegalArgumentException(
                    "value descriptor shape must be fully static");
        }
        if (value.descriptor().layout().isEmpty()) {
            throw new IllegalArgumentException(
                    "value descriptor layout must be resolved");
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("byteSize must be non-negative");
        }
        if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
            throw new IllegalArgumentException(
                    "byteAlignment must be a positive power of two");
        }
    }
}
