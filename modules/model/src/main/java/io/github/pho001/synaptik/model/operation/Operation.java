package io.github.pho001.synaptik.model.operation;

import java.util.Objects;

/**
 * Describes backend-independent computation semantics as an immutable pairing of a semantic kind
 * and typed attributes.
 *
 * <p>An operation is model-level, compile-time semantic state. It is not a graph node and does not
 * identify a computation occurrence; graph occurrence identity belongs to a node and its
 * {@code NodeId}, while {@code TensorId} and {@code ValueId} identify different domains. This
 * descriptor does not construct a graph, perform inference or compiler transformations, report
 * backend support or ownership, select an execution route, own storage or runtime state, or
 * provide any other executable behavior. A compiler may later consume this semantic description,
 * but runtime hot paths must not consume it.</p>
 *
 * <p>The descriptor validates only that both component references are present. It does not verify
 * that an attributes implementation belongs to a particular kind or validate family-specific
 * parameters; future operation-family contracts own those checks. Both references are retained
 * unchanged because their contracts require immutable values with stable equality and hashing.</p>
 *
 * <p>Record-generated equality and hashing use both typed components. Record-generated
 * {@link #toString()} text includes their diagnostic values, but that text is intended only for
 * inspection and is not a serialization, parsing, or dispatch contract.</p>
 *
 * @param kind the non-null semantic identity of the computation; the exact reference is stored
 *     unchanged
 * @param attrs the non-null typed semantic parameters; the exact reference is stored unchanged,
 *     and a parameterless kind uses {@link NoOperationAttrs#INSTANCE}
 */
public record Operation(OperationKind kind, OperationAttrs attrs) {
    /**
     * Creates an operation from its complete semantic kind and attribute state.
     *
     * <p>The supplied references are checked for null and then retained without copying,
     * normalization, replacement, or kind-to-attributes compatibility validation.</p>
     *
     * @param kind the non-null semantic identity of the computation; stored unchanged
     * @param attrs the non-null typed semantic parameters; stored unchanged, with
     *     {@link NoOperationAttrs#INSTANCE} representing no parameters
     * @throws NullPointerException if {@code kind} is {@code null}
     * @throws NullPointerException if {@code attrs} is {@code null}
     */
    public Operation {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(attrs, "attrs");
    }

    /**
     * Returns the semantic identity of the computation described by this operation.
     *
     * @return the exact non-null {@link OperationKind} object supplied at construction
     */
    public OperationKind kind() {
        return kind;
    }

    /**
     * Returns the typed semantic parameters that refine this operation's kind.
     *
     * @return the exact non-null {@link OperationAttrs} object supplied at construction;
     *     parameterless kinds return {@link NoOperationAttrs#INSTANCE} when it was supplied
     */
    public OperationAttrs attrs() {
        return attrs;
    }
}
