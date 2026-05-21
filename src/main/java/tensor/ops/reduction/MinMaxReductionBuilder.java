package tensor.ops.reduction;

import operations.Operation;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.layout.TensorLayoutTransform;

final class MinMaxReductionBuilder {
    private MinMaxReductionBuilder() {
    }

    static Tensor reduce(Tensor input, int dimension, boolean keepDims, boolean isMax) {
        ReductionShapeRules.requireFloatingInput(input, isMax ? "max" : "min");
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        int[] newShape = ReductionShapeRules.reduceShape(shape, normalizedDimension, keepDims);
        Operation op = isMax ? new reduceMax(normalizedDimension, keepDims) : new reduceMin(normalizedDimension, keepDims);
        Tensor out = TensorPrimitiveBuilder.unary(input, newShape, op,
                isMax ? "max_reduce" : "min_reduce", input.getDataType());
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor reducedForGrad = keepDims ? out : out.expandDims(normalizedDimension);
            Tensor outGradForGrad = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            Tensor grad = reduceMinMaxGrad(input, reducedForGrad, outGradForGrad, normalizedDimension);
            context.accumulate(input, grad);
        });
        return out;
    }

    static Tensor reduceAll(Tensor input, boolean isMax) {
        ReductionShapeRules.requireFloatingInput(input, isMax ? "max" : "min");
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                new int[]{1},
                isMax ? new reduceMax(-1) : new reduceMin(-1),
                isMax ? "max_reduce" : "min_reduce",
                input.getDataType()
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor grad = reduceMinMaxAllGrad(input, out, outGrad);
            context.accumulate(input, grad);
        });
        return out;
    }

    private static Tensor reduceMinMaxGrad(Tensor input, Tensor reducedKeepDims, Tensor outGradKeepDims, int dimension) {
        Tensor mask = input.equalTo(reducedKeepDims);
        Tensor maskNumeric = Tensor.where(mask, Tensor.onesLike(input), Tensor.zerosLike(input));
        Tensor winnerCount = maskNumeric.sum(dimension, true);
        Tensor scaledGrad = outGradKeepDims.div(winnerCount).expand(input.getShape());
        return Tensor.where(mask, scaledGrad, Tensor.zerosLike(input));
    }

    private static Tensor reduceMinMaxAllGrad(Tensor input, Tensor reduced, Tensor outGrad) {
        Tensor mask = input.equalTo(reduced);
        Tensor maskNumeric = Tensor.where(mask, Tensor.onesLike(input), Tensor.zerosLike(input));
        Tensor winnerCount = maskNumeric.sum();
        Tensor scaledGrad = outGrad.div(winnerCount).expand(input.getShape());
        return Tensor.where(mask, scaledGrad, Tensor.zerosLike(input));
    }
}
