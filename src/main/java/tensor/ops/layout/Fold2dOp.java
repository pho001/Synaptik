package tensor.ops.layout;

import operations.layout.fold2d;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.Window2dOptions;

/**
 * Graph-building definition for NCHW 2-D col2im accumulation.
 */
public final class Fold2dOp {
    private Fold2dOp() {
    }

    public static Tensor build(Tensor input, int[] outputShape, Window2dOptions options) {
        if (input == null) {
            throw new IllegalArgumentException("fold2d input cannot be null");
        }
        if (outputShape == null) {
            throw new IllegalArgumentException("fold2d outputShape cannot be null");
        }
        if (options == null) {
            throw new IllegalArgumentException("fold2d options cannot be null");
        }
        DataType dataType = input.getDataType();
        if (dataType == DataType.BOOL || dataType == DataType.INT32 || dataType == DataType.INT64) {
            throw new IllegalArgumentException("fold2d requires floating input.");
        }
        int[] normalizedOutputShape = outputShape.clone();
        Window2dShapeRules.validateFoldShapes(input.getShapeUnsafe(), normalizedOutputShape, options);
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                normalizedOutputShape,
                new fold2d(normalizedOutputShape, options),
                "fold2d",
                input.getDataType()
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.unfold2d(options));
        });
        return out;
    }
}
