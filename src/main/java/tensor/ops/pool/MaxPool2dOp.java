package tensor.ops.pool;

import operations.nn.pool.maxPool2d;
import operations.index.ScatterReduction;
import operations.reduction.ArgMaxTiePolicy;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.Pool2dOptions;
import tensor.options.Window2dOptions;

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
            int windowArea = Math.multiplyExact(options.kernelH(), options.kernelW());
            int windowCount = Math.multiplyExact(outH, outW);
            Window2dOptions windowOptions = windowOptions(options);
            Tensor columns = input.unfold2d(windowOptions)
                    .reshape(inputShape[0], inputShape[1], windowArea, windowCount);
            Tensor validColumns = Tensor.onesLike(input)
                    .unfold2d(windowOptions)
                    .reshape(inputShape[0], inputShape[1], windowArea, windowCount);
            Tensor maskedColumns = Tensor.where(
                    validColumns.greaterThan(Tensor.zerosLike(validColumns)),
                    columns,
                    negativeInfinityLike(columns)
            );
            Tensor winner = maskedColumns.argMax(2, true, ArgMaxTiePolicy.FIRST_INDEX);
            Tensor updates = outGrad.reshape(inputShape[0], inputShape[1], 1, windowCount);
            Tensor columnGrad = Tensor.zerosLike(columns)
                    .scatterElements(winner, updates, 2, ScatterReduction.NONE)
                    .reshape(inputShape[0], inputShape[1] * windowArea, windowCount);
            Tensor grad = columnGrad.fold2d(inputShape.clone(), windowOptions);
            context.accumulate(input, grad);
        });
        return out;
    }

    private static Window2dOptions windowOptions(Pool2dOptions options) {
        return new Window2dOptions(
                options.kernelH(),
                options.kernelW(),
                options.strideH(),
                options.strideW(),
                options.padH(),
                options.padW(),
                1,
                1,
                options.ceilMode()
        );
    }

    private static Tensor negativeInfinityLike(Tensor reference) {
        return Tensor.scalar(Double.NEGATIVE_INFINITY, reference.getDataType());
    }
}
