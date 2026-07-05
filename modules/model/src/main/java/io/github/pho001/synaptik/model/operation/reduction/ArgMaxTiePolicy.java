package io.github.pho001.synaptik.model.operation.reduction;

/**
 * Selects which logical index an arg-max reduction requests when several values share a maximum.
 *
 * <p>An equal maximum means that more than one value compares as the maximum under the later
 * execution contract. A logical index is the position along the selected tensor axis, independent
 * of physical storage offset or stride. This policy identifies the requested tie direction only;
 * it does not compare values, define NaN behavior, read storage, or execute a reduction.</p>
 *
 * <p>Enum identity provides immutable typed value semantics. Inherited enum text is diagnostic
 * only and is not a serialization, registry, dispatch, or backend contract.</p>
 */
public enum ArgMaxTiePolicy {
    /**
     * Requests the smallest logical index among values that share the maximum.
     *
     * <p>What constitutes an equal maximum, including NaN behavior, belongs to later numerical and
     * execution contracts.</p>
     */
    FIRST_INDEX,

    /**
     * Requests the largest logical index among values that share the maximum.
     *
     * <p>What constitutes an equal maximum, including NaN behavior, belongs to later numerical and
     * execution contracts.</p>
     */
    LAST_INDEX
}
