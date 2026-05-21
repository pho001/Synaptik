package tensor.ops.linalg;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for last-dimension linear projection.
 */
public final class LinearOp {
    private LinearOp() {
    }

    /**
     * Applies a linear projection without bias.
     *
     * <p>The input may have any rank {@code >= 2}. All leading dimensions are
     * treated as batch-like prefix dimensions and are preserved. The output shape
     * equals {@code input.shape} with the final dimension replaced by
     * {@code weight.shape[1]}.</p>
     *
     * @param input floating tensor shaped {@code [..., inFeatures]}; must be non-null
     * @param weight floating rank-2 tensor shaped {@code [inFeatures, outFeatures]}
     * @return projected tensor shaped {@code [..., outFeatures]}
     * @throws IllegalArgumentException if ranks, dimensions, or dtypes are invalid
     */
    public static Tensor build(Tensor input, Tensor weight) {
        LinearSpec spec = LinearSpec.resolve(input, weight, null);
        Operation op = spec.operation();
        Tensor out = TensorPrimitiveBuilder.binary(input, weight, spec.outShape(), op, "linear", spec.outputType());
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (input.getRequiresGrad()) {
                Tensor gradInput = outGrad.matmul(LinalgTensorRules.transposeLastTwoAxes(weight));
                context.accumulate(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = LinalgTensorRules.transposeLastTwoAxes(input).matmul(outGrad);
                context.accumulate(weight, LinalgTensorRules.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
        });
        return out;
    }

    /**
     * Applies a linear projection with bias.
     *
     * <p>Bias is broadcast over every leading input dimension and must be shaped
     * either {@code [outFeatures]} or {@code [1, outFeatures]}.</p>
     *
     * @param input floating tensor shaped {@code [..., inFeatures]}; must be non-null
     * @param weight floating rank-2 tensor shaped {@code [inFeatures, outFeatures]}
     * @param bias floating tensor shaped {@code [outFeatures]} or {@code [1, outFeatures]}
     * @return projected tensor plus broadcast bias
     * @throws IllegalArgumentException if ranks, dimensions, dtypes, or bias shape are invalid
     */
    public static Tensor build(Tensor input, Tensor weight, Tensor bias) {
        LinearSpec spec = LinearSpec.resolve(input, weight, bias);
        Operation op = spec.operation();
        Tensor out = TensorPrimitiveBuilder.ternary(input, weight, bias, spec.outShape(), op, "linear", spec.outputType());
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (input.getRequiresGrad()) {
                Tensor gradInput = outGrad.matmul(LinalgTensorRules.transposeLastTwoAxes(weight));
                context.accumulate(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = LinalgTensorRules.transposeLastTwoAxes(input).matmul(outGrad);
                context.accumulate(weight, LinalgTensorRules.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
            if (bias.getRequiresGrad()) {
                context.accumulate(bias, LinalgTensorRules.sumToShape(outGrad, bias.getShapeUnsafe()));
            }
        });
        return out;
    }
}
