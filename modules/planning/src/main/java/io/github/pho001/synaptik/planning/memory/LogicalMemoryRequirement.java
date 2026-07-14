package io.github.pho001.synaptik.planning.memory;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Describes the logical availability obligations for one value in a compiled graph.
 *
 * <p>The requirement retains the value's complete logical tensor descriptor, its optional
 * producing partition, every distinct consuming partition, and whether a graph-output boundary
 * must preserve it. Generated requirements derive that flag exactly from the owning graph's
 * declared outputs; directly constructed records retain the caller-supplied flag. These primitive
 * facts support overlapping classifications such as partition
 * input, partition output, cross-owner boundary, graph output, and partition-internal value
 * without storing a closed role taxonomy.</p>
 *
 * <p>This record is a compile-time recipe. It does not describe bytes, physical buffers, slots,
 * lifetimes, allocation, transfers, devices, backend representations, schedules, or runtime
 * residency. Record-generated equality and hashing compare all components by value, and the
 * generated {@link #toString()} is diagnostic text rather than a serialization format.</p>
 *
 * @param valueId non-null graph-local identity retained exactly from the graph value
 * @param descriptor non-null immutable logical tensor descriptor retained exactly from the graph
 *     value
 * @param producerPartition non-null value-based optional containing the exact supplied partition
 *     that produces the value, or empty for a graph input; optional-container identity is not
 *     promised
 * @param consumerPartitions non-null ordered distinct partitions that consume the value;
 *     membership is copied, elements must be non-null and unique by equality, and exact partition
 *     references are retained
 * @param graphOutput whether the value must remain available at a graph-output boundary; the
 *     package-private generator sets this to {@code true} exactly for a declared graph output
 */
public record LogicalMemoryRequirement(
        ValueId valueId,
        TensorDescriptor descriptor,
        Optional<PlannedPartition> producerPartition,
        List<PlannedPartition> consumerPartitions,
        boolean graphOutput) {
    /**
     * Creates an immutable logical requirement for one graph value.
     *
     * <p>Component references are validated in declaration order. Consumer partitions are then
     * scanned in encounter order for the first null or later equal duplicate before membership is
     * copied with {@link List#copyOf(java.util.Collection)}. The copy preserves exact element
     * references but does not retain the supplied list container.</p>
     *
     * @param valueId non-null graph-local value identity to retain exactly
     * @param descriptor non-null immutable logical tensor descriptor to retain exactly
     * @param producerPartition non-null value-based optional containing the exact producing
     *     partition, or empty when the value has no producing node
     * @param consumerPartitions non-null ordered distinct consuming partitions to snapshot;
     *     elements must be non-null and unique by equality
     * @param graphOutput whether the value must remain available at a graph-output boundary; this
     *     public constructor retains the supplied flag without an owning graph to validate it
     * @throws NullPointerException if {@code valueId}, {@code descriptor},
     *     {@code producerPartition}, or {@code consumerPartitions} is {@code null}; the message is
     *     the corresponding component name
     * @throws NullPointerException if a consumer partition is {@code null}; the message is
     *     {@code consumerPartitions[index]} with its zero-based encounter index
     * @throws IllegalArgumentException if a later consumer partition equals an earlier one; the
     *     message is {@code consumerPartitions[index] duplicates <partition>} with the later
     *     zero-based encounter index and duplicate partition's diagnostic text
     */
    public LogicalMemoryRequirement {
        Objects.requireNonNull(valueId, "valueId");
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(producerPartition, "producerPartition");
        Objects.requireNonNull(consumerPartitions, "consumerPartitions");

        var observedPartitions = new HashSet<PlannedPartition>();
        for (int index = 0; index < consumerPartitions.size(); index++) {
            PlannedPartition partition = Objects.requireNonNull(
                    consumerPartitions.get(index), "consumerPartitions[" + index + "]");
            if (!observedPartitions.add(partition)) {
                throw new IllegalArgumentException(
                        "consumerPartitions[" + index + "] duplicates " + partition);
            }
        }
        consumerPartitions = List.copyOf(consumerPartitions);
    }

    /**
     * Returns the graph-local identity of this logical value.
     *
     * @return the exact non-null immutable {@link ValueId} reference supplied at construction
     */
    @Override
    public ValueId valueId() {
        return valueId;
    }

    /**
     * Returns the value's complete backend-neutral logical tensor facts.
     *
     * @return the exact non-null immutable {@link TensorDescriptor} reference supplied at
     *     construction; no physical size or representation is implied
     */
    @Override
    public TensorDescriptor descriptor() {
        return descriptor;
    }

    /**
     * Returns the partition that produces this value, when one exists.
     *
     * @return non-null value-based optional containing the exact supplied producing-partition
     *     reference, or empty for a graph input; optional-container identity is not promised
     */
    @Override
    public Optional<PlannedPartition> producerPartition() {
        return producerPartition;
    }

    /**
     * Returns the immutable ordered snapshot of distinct partitions that consume this value.
     *
     * <p>The result follows supplied partition order, contains each equal partition at most once,
     * may be empty, and cannot be mutated. It retains exact partition element references without
     * promising identity with the original list container.</p>
     *
     * @return the non-null immutable ordered snapshot of distinct consuming partitions
     */
    @Override
    public List<PlannedPartition> consumerPartitions() {
        return consumerPartitions;
    }

    /**
     * Reports whether this value must remain available at a graph-output boundary.
     *
     * @return the stored boundary-obligation flag; generated requirements return {@code true}
     *     exactly for a value declared in the owning graph's output list
     */
    @Override
    public boolean graphOutput() {
        return graphOutput;
    }
}
