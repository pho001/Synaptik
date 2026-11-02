package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;

import java.util.function.Function;

/**
 * Routes ordering cotangents through one exact stable permutation or selected-index set.
 *
 * <p>A one-output {@code SORT} constructs exactly one matching public {@code ARGSORT}
 * occurrence after complete preflight and scatters the output cotangent through that permutation.
 * {@code TOP_K} instead consumes the original producer's retained canonical indices wrapper at
 * output slot one. Both use replacement scatter because the normalized logical indices are
 * unique. Equal keys, NaNs, signed zeros, cutoff membership, direction, and sorted-output order
 * therefore retain the current forward ordering convention without averaging. SORT's matching
 * ARGSORT is the sole forward-semantic recomputation in the closed task matrix; TOP_K never
 * recomputes its selected set. Both formulas create logical Tensor expressions only and make no
 * execution, sorting-algorithm, or backend claim.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class OrderingGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    OrderingGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds the sole selected input cotangent for one approved SORT or TOP_K values output.
     *
     * @param producer exact original ordering producer occurrence
     * @param gradient non-null accumulated values-output cotangent
     * @param primal non-null resolver from original primal wrappers to lexical values
     * @return a one-element input-aligned array containing the routed cotangent
     * @throws IllegalStateException if the occurrence is outside the approved ordering matrix
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Function<V, V> primal) {
        V input = producer.inputs().getFirst();
        V indices;
        int axis;
        if (producer.operation().kind() == OrderingKind.SORT) {
            SortAttrs attrs = (SortAttrs) producer.operation().attrs();
            indices = ops.argsort(primal.apply(input), attrs.axis(), attrs.descending());
            axis = attrs.axis();
        } else if (producer.operation().kind() == TopKKind.TOP_K) {
            TopKAttrs attrs = (TopKAttrs) producer.operation().attrs();
            indices = primal.apply(producer.output(1));
            axis = attrs.axis();
        } else {
            throw new IllegalStateException(
                    "ordering operation was not preflight-approved: " + producer.operation());
        }
        return ops.result(ops.scatterElements(ops.zeroLike(input), indices, gradient, axis, ScatterReduction.NONE));
    }
}
