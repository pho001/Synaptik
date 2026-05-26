package tensor.ops.layout;

import operations.layout.unfold2d;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.Window2dOptions;

/**
 * Graph-building definition for NCHW 2-D im2col materialization.
 */
public final class Unfold2dOp {
    private Unfold2dOp() {
    }

    public static Tensor build(Tensor input, Window2dOptions options) {
        if (options == null) {
            throw new IllegalArgumentException("unfold2d options cannot be null");
        }
        Window2dShapeRules.requireFloatingRank4(input, "unfold2d");
        int[] inputShape = input.getShapeUnsafe();
        int[] outShape = Window2dShapeRules.unfoldOutputShape(inputShape, options);
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                outShape,
                new unfold2d(options),
                "unfold2d",
                input.getDataType()
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, outGrad.fold2d(inputShape, options));
        });
        return out;
    }
}
