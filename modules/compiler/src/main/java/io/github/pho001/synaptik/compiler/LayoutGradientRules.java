package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds the closed logical layout-transform first-order formulas.
 *
 * <p>The selected rows are {@code CONTIGUOUS}, {@code RESHAPE}, {@code EXPAND},
 * {@code EXPAND_DIMS}, {@code SQUEEZE}, and {@code PERMUTE}. Formulas reverse only logical Shape
 * and axis metadata through public Tensor operations; they neither select physical aliasing or
 * materialization nor read storage, lower work, or execute computation. Preflight owns type,
 * Shape, attributes, and policy validation.</p>
 */
final class LayoutGradientRules {
    private LayoutGradientRules() {}

    /**
     * Builds the sole input cotangent for one approved layout occurrence.
     *
     * @param producer exact preflight-approved original logical layout-transform producer
     * @param gradient non-null accumulated cotangent for the producer's sole output
     * @return a new one-element array containing the exact input cotangent expression
     */
    static Tensor[] apply(TensorProducer producer, Tensor gradient) {
        Tensor input = producer.inputs().getFirst();
        if (producer.operation().kind() == ContiguousKind.CONTIGUOUS) {
            return new Tensor[] {gradient};
        }
        if (producer.operation().kind() == ShapeTransformKind.RESHAPE) {
            return new Tensor[] {gradient.reshape(input.descriptor().shape())};
        }
        if (producer.operation().kind() == ShapeTransformKind.EXPAND) {
            return new Tensor[] {gradient.sumToShape(input.descriptor().shape())};
        }
        AxisTransformKind kind = (AxisTransformKind) producer.operation().kind();
        if (kind == AxisTransformKind.EXPAND_DIMS) {
            int axis = ((AxisTransformAttrs) producer.operation().attrs()).axis();
            return new Tensor[] {gradient.squeeze(axis)};
        }
        if (kind == AxisTransformKind.SQUEEZE) {
            int axis = ((AxisTransformAttrs) producer.operation().attrs()).axis();
            return new Tensor[] {gradient.expandDims(axis)};
        }
        PermutationAttrs attrs = (PermutationAttrs) producer.operation().attrs();
        int[] inverse = new int[attrs.axes().size()];
        for (int outputAxis = 0; outputAxis < attrs.axes().size(); outputAxis++) {
            inverse[attrs.axes().get(outputAxis)] = outputAxis;
        }
        return new Tensor[] {gradient.permute(inverse)};
    }
}
