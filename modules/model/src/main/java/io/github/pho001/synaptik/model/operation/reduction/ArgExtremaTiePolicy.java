package io.github.pho001.synaptik.model.operation.reduction;

/**
 * Selects which logical index an arg-min or arg-max reduction requests among equal candidates.
 *
 * <p>A logical index is the coordinate along the selected tensor axis, independent of physical
 * storage, layout, stride, or traversal. For floating input, NaNs are preferred to non-NaNs,
 * multiple NaNs are ties, negative zero orders below positive zero, and infinities retain their
 * usual order. Integral input uses signed ordering. Thus the policy is applied only after the
 * minimum or maximum candidates are determined by that shared ordering.</p>
 *
 * <p>Enum identity provides immutable typed value semantics. This value stores no axis, Tensor,
 * Shape, input type, or traversal state and does not inspect values, execute a reduction, or
 * choose a compiler or backend route. Inherited enum text is diagnostic only rather than a
 * serialization, parser, registry, or dispatch contract.</p>
 */
public enum ArgExtremaTiePolicy {
    /**
     * Requests the smallest logical coordinate among equal minimum or maximum candidates.
     *
     * <p>The coordinate is independent of physical traversal order and storage layout.</p>
     */
    FIRST_INDEX,

    /**
     * Requests the largest logical coordinate among equal minimum or maximum candidates.
     *
     * <p>The coordinate is independent of physical traversal order and storage layout.</p>
     */
    LAST_INDEX
}
