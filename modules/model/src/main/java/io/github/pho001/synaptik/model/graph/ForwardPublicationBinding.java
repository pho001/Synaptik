package io.github.pho001.synaptik.model.graph;

import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.Objects;

/**
 * Associates one public tensor identity with one graph-local logical value identity.
 *
 * <p>This immutable data-transfer object carries model data used by the current compiler-owned
 * publication plan's ordered forward boundary. It remains separate from
 * {@link CompiledGraphModel}: the binding itself cannot prove that its {@link ValueId} belongs to
 * a particular graph. The compiler plan supplies that owning graph context. The binding retains
 * identities only, not a public tensor object, a gradient-publication binding, publication policy
 * or target, storage, backend state, or runtime publication behavior.</p>
 *
 * <p>Record-generated equality and hashing use both typed identities. The generated
 * {@link #toString()} is diagnostic text, not a serialization, parsing, global-identity, or
 * runtime-publication contract.</p>
 *
 * @param tensorId non-null identity from the public tensor lifecycle; the exact immutable
 *     reference is retained
 * @param valueId non-null graph-local logical value identity; the exact immutable reference is
 *     retained without an owning-graph assertion
 */
public record ForwardPublicationBinding(TensorId tensorId, ValueId valueId) {
    /**
     * Creates a standalone association between the two distinct identity domains.
     *
     * @param tensorId non-null public tensor identity to retain exactly
     * @param valueId non-null graph-local value identity to retain exactly
     * @throws NullPointerException if {@code tensorId} is {@code null}; the message is
     *     {@code tensorId}
     * @throws NullPointerException if {@code valueId} is {@code null}; the message is
     *     {@code valueId}
     */
    public ForwardPublicationBinding {
        Objects.requireNonNull(tensorId, "tensorId");
        Objects.requireNonNull(valueId, "valueId");
    }

    /**
     * Returns the public tensor identity represented by this standalone association.
     *
     * @return the exact non-null immutable {@link TensorId} reference supplied at construction;
     *     the result belongs to the public tensor identity domain
     */
    public TensorId tensorId() {
        return tensorId;
    }

    /**
     * Returns the identity of the logical graph value associated with the tensor identity.
     *
     * @return the exact non-null immutable {@link ValueId} reference supplied at construction;
     *     the result is graph-local and does not establish membership in a particular graph
     */
    public ValueId valueId() {
        return valueId;
    }
}
