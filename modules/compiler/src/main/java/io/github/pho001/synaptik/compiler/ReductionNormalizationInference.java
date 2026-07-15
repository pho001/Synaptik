package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;

/** Derives descriptors and constraints for reductions, scans, normalization, sorting, and top-K. */
final class ReductionNormalizationInference {
    private ReductionNormalizationInference() {}

    /**
     * Verifies one supported reduction, scan, normalization, or ordering occurrence.
     *
     * @param op non-null typed operation whose kind belongs to this family
     * @param in non-null ordered input descriptors supplied by the captured node
     * @return immutable derived descriptors and ordered candidate constraints; never {@code null}
     * @throws IllegalArgumentException if the kind is unsupported here or the occurrence violates
     *     its operand, attribute, Shape, or data-type contract
     */
    static CapturedGraphInference.InferenceResult infer(Operation op, List<TensorDescriptor> in) {
        if (op.kind() instanceof AggregateReductionKind k) return reduction(k, op.attrs(), in);
        if (op.kind() instanceof CumulativeScanKind) return scan((CumulativeScanAttrs) op.attrs(), in);
        if (op.kind() instanceof SoftmaxKind) return softmax((SoftmaxAttrs) op.attrs(), in);
        if (op.kind() instanceof LayerNormKind) return layerNorm(op.attrs(), in);
        if (op.kind() instanceof RmsNormKind) return rmsNorm((RmsNormAttrs) op.attrs(), in);
        if (op.kind() instanceof BatchNormKind k) return batchNorm(k, op.attrs(), in);
        if (op.kind() instanceof OrderingKind k) return ordering(k, (SortAttrs) op.attrs(), in);
        if (op.kind() instanceof TopKKind) return topK((TopKAttrs) op.attrs(), in);
        throw new IllegalArgumentException("unsupported reduction or normalization kind");
    }

    private static CapturedGraphInference.InferenceResult reduction(
            AggregateReductionKind kind, Object attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); List<CapturedGraphInference.ConstraintRequest> cs = new ArrayList<>();
        if (attrs instanceof SumToShapeAttrs sumTo) {
            requireNumericFor(kind, input); Shape target = sumTo.targetShape();
            if (target.rank() > input.shape().rank()) throw new IllegalArgumentException("sum-to target rank exceeds input rank");
            int offset = input.shape().rank() - target.rank();
            for (int axis = 0; axis < target.rank(); axis++) {
                Dimension t = target.dimension(axis), source = input.shape().dimension(axis + offset);
                cs.add(new CapturedGraphInference.ConstraintRequest("sum-to axis " + axis,
                        new AnyOf(List.of(new DimensionEqual(t, new StaticDimension(1)), new DimensionEqual(t, source)))));
            }
            return new CapturedGraphInference.InferenceResult(
                    List.of(ElementwiseInference.descriptor(input.dataType(), target, input.requiresGrad())), cs);
        }
        if (attrs instanceof MaskedReductionAttrs masked) {
            if (kind != AggregateReductionKind.SUM && kind != AggregateReductionKind.MEAN)
                throw new IllegalArgumentException("masked reduction supports SUM or MEAN");
            requireNumericFor(kind, input); ElementwiseInference.requireBool(in.get(1), "mask");
            ShapeBroadcast.broadcast(in.get(1).shape(), input.shape());
            requireNormalizedAxis(input.shape(), masked.axis());
            return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(
                    input.dataType(), reduce(input.shape(), List.of(masked.axis()), false), input.requiresGrad()));
        }
        List<Integer> axes; boolean keep;
        if (attrs == NoOperationAttrs.INSTANCE) { axes = allAxes(input.shape()); keep = false; }
        else if (attrs instanceof AxisReductionAttrs a) { requireNormalizedAxis(input.shape(), a.axis()); axes = List.of(a.axis()); keep = a.keepDimensions(); }
        else if (attrs instanceof MultiAxisReductionAttrs a) { validateAxes(input.shape(), a.axes()); axes = a.axes(); keep = a.keepDimensions(); }
        else if (attrs instanceof StatisticalReductionAttrs a) {
            validateAxes(input.shape(), a.axes()); axes = a.axes(); keep = a.keepDimensions();
            ElementwiseInference.requireFloating(input, "input");
            Shape domain = select(input.shape(), axes);
            cs.add(new CapturedGraphInference.ConstraintRequest("statistical reduction domain",
                    new ShapeElementCountValue(domain, Math.addExact(a.correction(), 1), ShapeElementCountValue.Comparison.AT_LEAST)));
        } else if (attrs instanceof ArgExtremaAttrs a) {
            requireNormalizedAxis(input.shape(), a.axis()); ElementwiseInference.requireNumeric(input, "input");
            cs.add(new CapturedGraphInference.ConstraintRequest("arg-extrema selected extent",
                    new DimensionAtLeast(input.shape().dimension(a.axis()), 1)));
            TensorDescriptor out = ElementwiseInference.descriptor(DataType.INT64,
                    reduce(input.shape(), List.of(a.axis()), a.keepDimensions()), false);
            return new CapturedGraphInference.InferenceResult(List.of(out), cs);
        } else throw new IllegalArgumentException("unsupported reduction attributes");
        requireInput(kind, input);
        DataType resultType = kind == AggregateReductionKind.ARG_MAX || kind == AggregateReductionKind.ARG_MIN
                ? DataType.INT64 : input.dataType();
        boolean grad = resultType.isFloating() && input.requiresGrad();
        return new CapturedGraphInference.InferenceResult(
                List.of(ElementwiseInference.descriptor(resultType, reduce(input.shape(), axes, keep), grad)), cs);
    }

    private static CapturedGraphInference.InferenceResult scan(CumulativeScanAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); ElementwiseInference.requireNumeric(input, "input");
        requireNormalizedAxis(input.shape(), attrs.axis());
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(
                input.dataType(), input.shape(), input.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult softmax(SoftmaxAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); ElementwiseInference.requireFloating(input, "input"); requireNormalizedAxis(input.shape(), attrs.axis());
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(input.dataType(), input.shape(), input.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult layerNorm(Object attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); ElementwiseInference.requireFloating(input, "input");
        Shape normalized; DataType type = input.dataType(); boolean grad = input.requiresGrad();
        if (attrs instanceof LayerNormAttrs a) { normalized = a.normalizedShape(); requireScalarType(a.epsilon().dataType(), type, "epsilon"); }
        else {
            AffineLayerNormAttrs a = (AffineLayerNormAttrs) attrs; normalized = a.normalizedShape();
            TensorDescriptor scale = in.get(1), bias = in.get(2); ElementwiseInference.requireFloating(scale, "scale"); ElementwiseInference.requireFloating(bias, "bias");
            requireShape(scale.shape(), normalized, "scale"); requireShape(bias.shape(), normalized, "bias");
            type = DataTypePromotion.promoteFloating(type, scale.dataType()); type = DataTypePromotion.promoteFloating(type, bias.dataType());
            grad |= scale.requiresGrad() || bias.requiresGrad(); requireScalarType(a.epsilon().dataType(), type, "epsilon");
        }
        List<CapturedGraphInference.ConstraintRequest> constraints = trailing(input.shape(), normalized, "normalized");
        return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(type, input.shape(), grad)), constraints);
    }

    private static CapturedGraphInference.InferenceResult rmsNorm(RmsNormAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); ElementwiseInference.requireFloating(input, "input");
        DataType type = input.dataType(); boolean grad = input.requiresGrad();
        if (in.size() == 2) { TensorDescriptor weight = in.get(1); ElementwiseInference.requireFloating(weight, "weight"); requireShape(weight.shape(), attrs.normalizedShape(), "weight"); type = DataTypePromotion.promoteFloating(type, weight.dataType()); grad |= weight.requiresGrad(); }
        requireScalarType(attrs.epsilon().dataType(), type, "epsilon");
        return new CapturedGraphInference.InferenceResult(
                List.of(ElementwiseInference.descriptor(type, input.shape(), grad)), trailing(input.shape(), attrs.normalizedShape(), "normalized"));
    }

    private static CapturedGraphInference.InferenceResult batchNorm(BatchNormKind kind, Object attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); ElementwiseInference.requireFloating(input, "input");
        int axis = kind == BatchNormKind.BATCH_NORM_INFERENCE
                ? ((BatchNormInferenceAttrs) attrs).channelAxis() : ((BatchNormTrainingAttrs) attrs).channelAxis();
        requireNormalizedAxis(input.shape(), axis); if (input.shape().rank() < 2) throw new IllegalArgumentException("input rank must be at least 2");
        DataType type = input.dataType(); boolean allGrad = input.requiresGrad(); List<CapturedGraphInference.ConstraintRequest> cs = new ArrayList<>();
        Dimension channel = input.shape().dimension(axis); Shape statistic = Shape.ofDimensions(channel);
        for (int i = 1; i < 5; i++) {
            TensorDescriptor operand = in.get(i); ElementwiseInference.requireFloating(operand, "input[" + i + "]");
            if (operand.shape().rank() != 1) throw new IllegalArgumentException("batch-normalization vector rank must be one");
            type = DataTypePromotion.promoteFloating(type, operand.dataType()); allGrad |= operand.requiresGrad();
            cs.add(new CapturedGraphInference.ConstraintRequest("channel input[" + i + "]", new DimensionEqual(channel, operand.shape().dimension(0))));
        }
        if (kind == BatchNormKind.BATCH_NORM_INFERENCE) {
            requireScalarType(((BatchNormInferenceAttrs) attrs).epsilon().dataType(), type, "epsilon");
            return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(type, input.shape(), allGrad)), cs);
        }
        BatchNormTrainingAttrs a = (BatchNormTrainingAttrs) attrs; requireScalarType(a.momentum().dataType(), type, "momentum"); requireScalarType(a.epsilon().dataType(), type, "epsilon");
        boolean inputGrad = input.requiresGrad();
        List<TensorDescriptor> outputs = List.of(
                ElementwiseInference.descriptor(type, input.shape(), inputGrad || in.get(1).requiresGrad() || in.get(2).requiresGrad()),
                ElementwiseInference.descriptor(type, statistic, inputGrad || in.get(3).requiresGrad()),
                ElementwiseInference.descriptor(type, statistic, inputGrad || in.get(4).requiresGrad()),
                ElementwiseInference.descriptor(type, statistic, inputGrad),
                ElementwiseInference.descriptor(type, statistic, inputGrad));
        Shape domain = withoutAxis(input.shape(), axis);
        cs.add(new CapturedGraphInference.ConstraintRequest("batch-normalization training domain",
                new AnyOf(List.of(new DimensionEqual(channel, new StaticDimension(0)),
                        new ShapeElementCountValue(domain, 2, ShapeElementCountValue.Comparison.AT_LEAST)))));
        return new CapturedGraphInference.InferenceResult(outputs, cs);
    }

    private static CapturedGraphInference.InferenceResult ordering(OrderingKind kind, SortAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); requireNormalizedAxis(input.shape(), attrs.axis()); boolean argsort = kind == OrderingKind.ARGSORT;
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(
                argsort ? DataType.INT64 : input.dataType(), input.shape(), !argsort && input.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult topK(TopKAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); requireNormalizedAxis(input.shape(), attrs.axis());
        Dimension[] dims = input.shape().dimensions().toArray(Dimension[]::new); Dimension selected = dims[attrs.axis()]; dims[attrs.axis()] = new StaticDimension(attrs.k()); Shape shape = Shape.ofDimensions(dims);
        var c = new CapturedGraphInference.ConstraintRequest("top-K selected extent", new DimensionAtLeast(selected, attrs.k()));
        return CapturedGraphInference.InferenceResult.constrained(List.of(
                ElementwiseInference.descriptor(input.dataType(), shape, input.requiresGrad()),
                ElementwiseInference.descriptor(DataType.INT64, shape, false)), c);
    }

    private static void requireInput(AggregateReductionKind kind, TensorDescriptor d) {
        if (kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY) ElementwiseInference.requireBool(d, "input");
        else if (kind == AggregateReductionKind.MEAN || kind == AggregateReductionKind.LOG_SUM_EXP
                || kind == AggregateReductionKind.VARIANCE || kind == AggregateReductionKind.STANDARD_DEVIATION
                || kind == AggregateReductionKind.L1_NORM || kind == AggregateReductionKind.L2_NORM) ElementwiseInference.requireFloating(d, "input");
        else ElementwiseInference.requireNumeric(d, "input");
    }
    private static void requireNumericFor(AggregateReductionKind kind, TensorDescriptor d) { requireInput(kind, d); }
    private static void requireNormalizedAxis(Shape shape, int axis) { if (axis < 0 || axis >= shape.rank()) throw new IllegalArgumentException("axis is not normalized for input rank"); }
    private static void validateAxes(Shape shape, List<Integer> axes) { for (int axis : axes) requireNormalizedAxis(shape, axis); }
    private static List<Integer> allAxes(Shape shape) { List<Integer> result = new ArrayList<>(); for (int i=0;i<shape.rank();i++) result.add(i); return result; }
    private static Shape reduce(Shape shape, List<Integer> axes, boolean keep) {
        List<Dimension> result = new ArrayList<>();
        for (int i=0;i<shape.rank();i++) { if (axes.contains(i)) { if (keep) result.add(new StaticDimension(1)); } else result.add(shape.dimension(i)); }
        return Shape.ofDimensions(result.toArray(Dimension[]::new));
    }
    private static Shape select(Shape shape, List<Integer> axes) { return Shape.ofDimensions(axes.stream().map(shape::dimension).toArray(Dimension[]::new)); }
    private static Shape withoutAxis(Shape shape, int axis) { List<Integer> axes=allAxes(shape); axes.remove(Integer.valueOf(axis)); return select(shape,axes); }
    private static List<CapturedGraphInference.ConstraintRequest> trailing(Shape input, Shape normalized, String subject) {
        if (normalized.rank() > input.rank()) throw new IllegalArgumentException("normalized rank exceeds input rank");
        List<CapturedGraphInference.ConstraintRequest> result = new ArrayList<>(); int offset=input.rank()-normalized.rank();
        for(int i=0;i<normalized.rank();i++) result.add(new CapturedGraphInference.ConstraintRequest(subject+" axis "+i,new DimensionEqual(input.dimension(offset+i),normalized.dimension(i))));
        return result;
    }
    private static void requireShape(Shape actual, Shape expected, String role) { if (!actual.equals(expected)) throw new IllegalArgumentException(role+" shape mismatch"); }
    private static void requireScalarType(DataType scalar, DataType result, String role) { if (scalar != result) throw new IllegalArgumentException(role+" data type must match result"); }
}
