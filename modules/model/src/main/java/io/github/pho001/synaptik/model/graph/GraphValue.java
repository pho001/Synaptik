package io.github.pho001.synaptik.model.graph;

import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Objects;

/**
 * Describes one immutable logical data value within an owning graph context.
 *
 * <p>A graph value is data that may be a graph input, an intermediate result, or a graph output.
 * It is distinct from a {@link CompiledNode}, which represents a computation occurrence, and from
 * the planned public mutable {@code Tensor} API object used to build expressions. It also does not
 * represent physical storage, a prepared memory slot, a device buffer, or runtime residency.</p>
 *
 * <p>Producer and consumer relationships are deliberately absent. A graph input has no producer,
 * and an owning graph can derive producer and use relationships from node input and output lists
 * without duplicating them in this value. Graph-wide existence, uniqueness, topology, descriptor
 * agreement, and role classification therefore belong to the owning
 * {@link CompiledGraphModel}. This record supplies model data only; by itself it performs no graph
 * capture, compiler transformation, or graph-wide validation.</p>
 *
 * <p>Both components are immutable values and their exact references are retained. Record-generated
 * equality and hashing use both components. The generated {@link #toString()} exposes diagnostic
 * component values but is not a serialization or parsing format.</p>
 *
 * @param id non-null graph-local identity of this logical value; the exact immutable reference is
 *     retained, and uniqueness is interpreted only within an owning graph
 * @param descriptor non-null immutable description of the value's logical tensor facts; the exact
 *     reference is retained without adding storage, execution, or backend state
 */
public record GraphValue(ValueId id, TensorDescriptor descriptor) {
    /**
     * Creates a logical graph value from its graph-local identity and tensor description.
     *
     * <p>The constructor checks only that both immutable component references are present and then
     * retains them unchanged. It does not find a producer, classify the value as an input or
     * output, allocate storage, or validate the value against any node or owning graph.</p>
     *
     * @param id non-null graph-local value identity to retain exactly; allocation and uniqueness
     *     belong to the owning graph construction lifecycle
     * @param descriptor non-null immutable logical tensor description to retain exactly
     * @throws NullPointerException if {@code id} is {@code null}; the message is {@code id}
     * @throws NullPointerException if {@code descriptor} is {@code null}; the message is
     *     {@code descriptor}
     */
    public GraphValue {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(descriptor, "descriptor");
    }

    /**
     * Returns this value's identity within its owning graph context.
     *
     * @return the exact non-null immutable {@link ValueId} reference supplied at construction;
     *     its scope is graph-local and the result does not identify storage or a public tensor
     */
    public ValueId id() {
        return id;
    }

    /**
     * Returns the logical tensor description associated with this graph value.
     *
     * @return the exact non-null immutable {@link TensorDescriptor} reference supplied at
     *     construction, without storage, graph relationships, or runtime state
     */
    public TensorDescriptor descriptor() {
        return descriptor;
    }
}
