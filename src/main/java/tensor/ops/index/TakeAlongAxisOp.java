package tensor.ops.index;

import operations.index.ScatterReduction;
import operations.index.takeAlongAxis;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for output-shaped {@code takeAlongAxis}.
 */
public final class TakeAlongAxisOp {
    private TakeAlongAxisOp() {
    }

    public static Tensor build(Tensor input, Tensor indices, int dimension) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("takeAlongAxis inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("takeAlongAxis indices must be numeric integral values.");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        IndexSupport.validateTakeAlongAxisShape(inputShape, indices.getShape(), normalizedDimension);

        Tensor out = TensorPrimitiveBuilder.binary(
                input,
                indices,
                indices.getShape().clone(),
                new takeAlongAxis(normalizedDimension),
                "takeAlongAxis",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = Tensor.zerosLike(input).scatterElements(indices, outGrad, normalizedDimension, ScatterReduction.ADD);
            context.accumulate(input, grad);
        });
        return out;
    }
}
