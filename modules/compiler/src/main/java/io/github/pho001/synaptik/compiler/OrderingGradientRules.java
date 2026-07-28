package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

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
 */
final class OrderingGradientRules {
    private OrderingGradientRules() {}

    /**
     * Builds the sole selected input cotangent for one approved SORT or TOP_K values output.
     *
     * @param producer exact original ordering producer occurrence
     * @param gradient non-null accumulated values-output cotangent
     * @param constants non-null request-local exact typed positive-zero owner
     * @return a one-element input-aligned array containing the routed cotangent
     * @throws IllegalStateException if the occurrence is outside the approved ordering matrix
     */
    static Tensor[] apply(
            TensorProducer producer,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        Tensor indices;
        int axis;
        if (producer.operation().kind() == OrderingKind.SORT) {
            SortAttrs attrs = (SortAttrs) producer.operation().attrs();
            indices = input.argsort(attrs.axis(), attrs.descending());
            axis = attrs.axis();
        } else if (producer.operation().kind() == TopKKind.TOP_K) {
            TopKAttrs attrs = (TopKAttrs) producer.operation().attrs();
            indices = producer.output(1);
            axis = attrs.axis();
        } else {
            throw new IllegalStateException(
                    "ordering operation was not preflight-approved: " + producer.operation());
        }
        return new Tensor[] {
            constants.zeroLike(input)
                    .scatterElements(indices, gradient, axis, ScatterReduction.NONE)
        };
    }
}
