package tensor.ops.linalg;

import graph.optimizer.intent.BackendIntentPropagator;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Last-dimension linear projection operations.
 *
 * <p>These methods build differentiable graph tensors for the generic tensor
 * operation {@code input[..., inFeatures] * weight[inFeatures, outFeatures]}.
 * They intentionally do not model a neural-network layer: ownership of parameter
 * initialization, layer state, and model structure belongs in a consumer framework.
 * Inputs are not mutated.</p>
 */
public final class TensorLinearOps {
    private TensorLinearOps() {
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
    public static Tensor linear(Tensor input, Tensor weight) {
        LinearSpec spec = LinearSpec.resolve(input, weight, null);
        Operation op = spec.operation();
        Tensor out = TensorPrimitiveBuilder.binary(input, weight, spec.outShape(), op, "linear", spec.outputType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (input.getRequiresGrad()) {
                Tensor gradInput = outGrad.matmul(LinalgSupport.transposeLastTwoAxes(weight));
                BackendIntentPropagator.preserve(gradInput, out);
                LinalgSupport.accumulateGradient(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = LinalgSupport.transposeLastTwoAxes(input).matmul(outGrad);
                BackendIntentPropagator.preserve(gradWeight, out);
                LinalgSupport.accumulateGradient(weight, LinalgSupport.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
        });
        return out;
    }

    /**
     * Applies a linear projection with bias.
     *
     * <p>Bias is broadcast over every leading input dimension and must be shaped
     * either {@code [outFeatures]} or {@code [1, outFeatures]}. For example,
     * input {@code [batch, time, inFeatures]} and weight
     * {@code [inFeatures, outFeatures]} produce output
     * {@code [batch, time, outFeatures]}.</p>
     *
     * @param input floating tensor shaped {@code [..., inFeatures]}; must be non-null
     * @param weight floating rank-2 tensor shaped {@code [inFeatures, outFeatures]}
     * @param bias floating tensor shaped {@code [outFeatures]} or {@code [1, outFeatures]}
     * @return projected tensor plus bias shaped {@code [..., outFeatures]}
     * @throws IllegalArgumentException if ranks, dimensions, dtypes, or bias shape are invalid
     */
    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        LinearSpec spec = LinearSpec.resolve(input, weight, bias);
        Operation op = spec.operation();
        Tensor out = TensorPrimitiveBuilder.ternary(input, weight, bias, spec.outShape(), op, "linear", spec.outputType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (input.getRequiresGrad()) {
                Tensor gradInput = outGrad.matmul(LinalgSupport.transposeLastTwoAxes(weight));
                BackendIntentPropagator.preserve(gradInput, out);
                LinalgSupport.accumulateGradient(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = LinalgSupport.transposeLastTwoAxes(input).matmul(outGrad);
                BackendIntentPropagator.preserve(gradWeight, out);
                LinalgSupport.accumulateGradient(weight, LinalgSupport.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
            if (bias.getRequiresGrad()) {
                LinalgSupport.accumulateGradient(bias, LinalgSupport.sumToShape(outGrad, bias.getShapeUnsafe()));
            }
        });
        return out;
    }
}
