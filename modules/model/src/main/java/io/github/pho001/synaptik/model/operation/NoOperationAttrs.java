package io.github.pho001.synaptik.model.operation;

/**
 * Represents the complete attribute value for an operation kind with no semantic parameters.
 *
 * <p>This enum is an immutable canonical singleton. Operations without parameters use
 * {@link #INSTANCE} instead of {@code null}, an empty map, or a newly allocated placeholder. Enum
 * identity therefore also supplies stable equality, hashing, and diagnostic text. A
 * parameterless kind explicitly names this exact class in its family-owned
 * {@link OperationSignature}; the value is not a permissive fallback for a kind that expects
 * typed parameters.</p>
 *
 * <p>The value carries no operation-family, backend, planning, execution, storage, or runtime
 * metadata.</p>
 */
public enum NoOperationAttrs implements OperationAttrs {
    /**
     * The canonical non-null attribute value for an operation kind with no semantic parameters.
     *
     * <p>The value is immutable and carries no hidden parameters. Its enum name also provides the
     * stable diagnostic text {@code INSTANCE}; callers use its singleton identity or enum value
     * equality and never allocate an alternative empty attribute object.</p>
     */
    INSTANCE
}
