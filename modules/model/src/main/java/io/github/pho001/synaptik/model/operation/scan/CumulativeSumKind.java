package io.github.pho001.synaptik.model.operation.scan;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent cumulative-sum scan semantics.
 *
 * <p>A cumulative sum has one logical input and produces one output position for every input
 * position. Unlike an aggregate reduction, it therefore preserves the input shape. The scan axis,
 * inclusion mode, and traversal direction are carried by {@link CumulativeSumAttrs}; this enum
 * stores none of those parameters, input state, result state, or graph-occurrence identity.</p>
 *
 * <p>The valid family composition pairs {@link #CUM_SUM} with {@link CumulativeSumAttrs}. The
 * generic {@link io.github.pho001.synaptik.model.operation.Operation Operation} descriptor checks
 * only that its kind and attributes are non-null and does not enforce this family-specific
 * pairing.</p>
 *
 * <p>This vocabulary describes requested mathematics only. It does not define eligible data
 * types, result descriptors, accumulation precision, numerical edge cases, gradients, value
 * execution, storage, compiler behavior, or backend availability. Enum identity supplies typed
 * equality and hashing. Inherited {@link #name()} and {@link #toString()} text is diagnostic only,
 * not a serialization, parsing, registry, dispatch, or kernel contract.</p>
 */
public enum CumulativeSumKind implements OperationKind {
    /**
     * Requests cumulative addition along the normalized axis in {@link CumulativeSumAttrs}.
     *
     * <p>The attributes select inclusive or exclusive output and forward or reverse traversal.
     * Output positions remain in input order in every mode: reverse traversal changes which
     * values contribute to each position and does not reverse the output. For logical input
     * {@code [1, 2, 3]}, the inclusive-forward, exclusive-forward, inclusive-reverse, and
     * exclusive-reverse meanings are respectively {@code [1, 3, 6]}, {@code [0, 1, 3]},
     * {@code [6, 5, 3]}, and {@code [5, 3, 0]}. The zero is the additive identity emitted at the
     * first position visited by an exclusive scan.</p>
     *
     * <p>Input eligibility, shape and result construction, numerical policy, gradients, execution,
     * and backend support belong to later owning contracts.</p>
     */
    CUM_SUM
}
