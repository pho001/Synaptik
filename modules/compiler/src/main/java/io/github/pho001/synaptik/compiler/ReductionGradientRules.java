package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Builds the closed {@code SUM} and {@code CUM_SUM} first-order formulas.
 *
 * <p>Full, one-axis, and ordered multi-axis sums restore removed axes before expanding to the
 * exact input Shape; masked sum additionally routes the restored cotangent through its BOOL mask,
 * and the locally proved sum-to-Shape variant expands directly to the source Shape. An empty
 * multi-axis selection passes the cotangent through. Cumulative sum preserves the normalized axis
 * and exclusivity while reversing the forward direction flag. Preflight owns type, Shape,
 * attributes, role, and policy validation.</p>
 *
 * <p>More precisely, masked SUM inserts the removed axis into {@code g}, expands it to the data
 * Shape, and returns {@code where(mask, restored, Z_data)} for the data role plus no mask
 * cotangent. A shape-target SUM returns {@code g.expand(input.shape())} only after preflight has
 * proved that every right-aligned target Dimension is the exact input Dimension or static one;
 * leading input axes are allowed, while binding-dependent pairs are rejected.</p>
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
     * @param constants non-null request-local owner of exact typed positive-zero splats
     * @return a new input-position-aligned array containing selected cotangent expressions and
     *     {@code null} for the non-differentiable masked-SUM mask role
     */
    static Tensor[] apply(
            TensorProducer producer,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        if (producer.operation().kind() == CumulativeScanKind.CUM_SUM) {
            CumulativeScanAttrs attrs = (CumulativeScanAttrs) producer.operation().attrs();
            return new Tensor[] {
                gradient.cumSum(attrs.axis(), attrs.exclusive(), !attrs.reverse())
            };
        }
        if (producer.operation().attrs() instanceof SumToShapeAttrs) {
            return new Tensor[] {gradient.expand(input.descriptor().shape())};
        }
        if (producer.operation().attrs() instanceof MaskedReductionAttrs attrs) {
            Tensor restored = gradient.expandDims(attrs.axis())
                    .expand(input.descriptor().shape());
            return new Tensor[] {
                Tensor.where(
                        producer.inputs().get(1),
                        restored,
                        constants.zeroLike(input)),
                null
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
