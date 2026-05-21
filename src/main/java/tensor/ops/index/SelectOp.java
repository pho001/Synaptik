package tensor.ops.index;

import operations.layout.select;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for axis {@code select}.
 */
public final class SelectOp {
    private SelectOp() {
    }

    public static Tensor build(Tensor input, int dimension, int index) {
        if (input == null) {
            throw new IllegalArgumentException("select input cannot be null");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        int normalizedIndex = IndexShapeRules.normalizeIndex(index, inputShape[normalizedDimension]);
        int[] outShape = IndexShapeRules.reduceShape(inputShape, normalizedDimension);
        int[] outStrides = IndexShapeRules.reduceStrides(input.getStridesUnsafe(), normalizedDimension);
        int outStorageOffset = input.getStorageOffsetUnsafe() + normalizedIndex * input.getStridesUnsafe()[normalizedDimension];

        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                outShape,
                outStrides,
                outStorageOffset,
                new select(normalizedDimension, normalizedIndex),
                "select",
                input.getDataType()
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor zeroBase = Tensor.zerosLike(input);
            Tensor indices = IndexShapeRules.constantIndexTensor(IndexShapeRules.reduceShape(input.getShapeUnsafe(), normalizedDimension), normalizedIndex);
            Tensor grad = zeroBase.scatterAdd(indices, outGrad, normalizedDimension);
            context.accumulate(input, grad);
        });
        return out;
    }
}
