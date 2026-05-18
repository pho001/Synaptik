package tensor.ops.pool;

import operations.nn.pool.avgPool2d;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;
import tensor.options.Pool2dOptions;

/**
 * Graph-building definition for NCHW 2-D average pooling.
 */
public final class AvgPool2dOp {
    private AvgPool2dOp() {
    }

    /**
     * Applies average pooling over spatial dimensions.
     *
     * @param input rank-4 input tensor with shape {@code [N, C, H, W]}
     * @param options pooling options; must be non-null
     * @return pooled tensor with shape {@code [N, C, outH, outW]}
     * @throws IllegalArgumentException if input/options are null, input is not a
     *                                  rank-4 floating tensor, or window geometry is invalid
     */
    public static Tensor build(Tensor input, Pool2dOptions options) {
        PoolSupport.validateInput(input, options, "avgPool2d");
        int[] inputShape = input.getShapeUnsafe();
        int outH = PoolSupport.inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), options.ceilMode(), "height");
        int outW = PoolSupport.inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), options.ceilMode(), "width");
        PoolSupport.validateWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), outH, "height");
        PoolSupport.validateWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), outW, "width");

        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                new int[]{inputShape[0], inputShape[1], outH, outW},
                new avgPool2d(options),
                "avgPool2d",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = TensorPrimitiveBuilder.unaryNoGrad(
                    outGrad,
                    inputShape.clone(),
                    new operations.nn.pool.avgPool2dBackwardInput(options, inputShape),
                    "avgPool2dBackwardInput",
                    input.getDataType()
            );
            PoolSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
