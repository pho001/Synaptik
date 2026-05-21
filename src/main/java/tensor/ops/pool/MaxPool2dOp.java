package tensor.ops.pool;

import operations.nn.pool.maxPool2d;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.Pool2dOptions;

/**
 * Graph-building definition for NCHW 2-D max pooling.
 */
public final class MaxPool2dOp {
    private MaxPool2dOp() {
    }

    /**
     * Applies max pooling over spatial dimensions.
     *
     * @param input rank-4 input tensor with shape {@code [N, C, H, W]}
     * @param options pooling options; must be non-null
     * @return pooled tensor with shape {@code [N, C, outH, outW]}
     * @throws IllegalArgumentException if input/options are null, input is not a
     *                                  rank-4 floating tensor, or window geometry is invalid
     */
    public static Tensor build(Tensor input, Pool2dOptions options) {
        Pool2dShapeRules.validateInput(input, options, "maxPool2d");
        int[] inputShape = input.getShapeUnsafe();
        int outH = Pool2dShapeRules.inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), options.ceilMode(), "height");
        int outW = Pool2dShapeRules.inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), options.ceilMode(), "width");
        Pool2dShapeRules.validateWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), outH, "height");
        Pool2dShapeRules.validateWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), outW, "width");

        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                new int[]{inputShape[0], inputShape[1], outH, outW},
                new maxPool2d(options),
                "maxPool2d",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                    outGrad,
                    input,
                    inputShape.clone(),
                    new operations.nn.pool.maxPool2dBackwardInput(options, inputShape),
                    "maxPool2dBackwardInput",
                    input.getDataType()
            );
            context.accumulate(input, grad);
        });
        return out;
    }
}
