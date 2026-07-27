package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Builds the closed ordinary {@code SUM} and {@code CUM_SUM} first-order formulas.
 *
 * <p>Full, one-axis, and ordered multi-axis sums restore removed axes before expanding to the
 * exact input Shape; an empty multi-axis selection passes the cotangent through. Cumulative sum
 * preserves the normalized axis and exclusivity while reversing the forward direction flag.
 * Preflight owns type, Shape, attributes, role, and policy validation.</p>
 *
 * <p>The formulas use public Tensor operations only and do not evaluate values, read storage,
 * capture a graph, lower work, or execute computation.</p>
 */
final class ReductionGradientRules {
    private ReductionGradientRules() {}

    /**
     * Builds the sole input cotangent for one approved reduction or scan.
     *
     * @param producer exact preflight-approved original {@code SUM} or {@code CUM_SUM} producer
     * @param gradient non-null accumulated cotangent for the producer's sole output
     * @return a new one-element array containing the exact input cotangent expression
     */
    static Tensor[] apply(TensorProducer producer, Tensor gradient) {
        Tensor input = producer.inputs().getFirst();
        if (producer.operation().kind() == CumulativeScanKind.CUM_SUM) {
            CumulativeScanAttrs attrs = (CumulativeScanAttrs) producer.operation().attrs();
            return new Tensor[] {
                gradient.cumSum(attrs.axis(), attrs.exclusive(), !attrs.reverse())
            };
        }

        Tensor restored = gradient;
        if (producer.operation().attrs() == NoOperationAttrs.INSTANCE) {
            for (int axis = 0; axis < input.descriptor().shape().rank(); axis++) {
                restored = restored.expandDims(axis);
            }
        } else if (producer.operation().attrs() instanceof AxisReductionAttrs attrs) {
            if (!attrs.keepDimensions()) {
                restored = restored.expandDims(attrs.axis());
            }
        } else {
            MultiAxisReductionAttrs attrs =
                    (MultiAxisReductionAttrs) producer.operation().attrs();
            if (attrs.axes().isEmpty()) {
                return new Tensor[] {gradient};
            }
            if (!attrs.keepDimensions()) {
                List<Integer> axes = new ArrayList<>(attrs.axes());
                axes.sort(Comparator.naturalOrder());
                for (int axis : axes) {
                    restored = restored.expandDims(axis);
                }
            }
        }
        return new Tensor[] {restored.expand(input.descriptor().shape())};
    }
}
