package tensor.ops.linalg;

import graph.optimizer.intent.BackendIntentPropagator;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Fully connected layer operations.
 *
 * <p>These methods build differentiable graph tensors for {@code input * weight}
 * with an optional broadcast-compatible bias. Inputs are not mutated.</p>
 */
public final class TensorLinearOps {
    private TensorLinearOps() {
    }

    /**
     * Applies a linear projection without bias.
     *
     * @param input input activations; must be non-null and floating numeric
     * @param weight weight matrix or batched weights accepted by {@link LinearSpec}
     * @return projected tensor
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
     * @param input input activations; must be non-null and floating numeric
     * @param weight weight matrix or batched weights accepted by {@link LinearSpec}
     * @param bias bias tensor broadcast-compatible with the output; must be non-null
     * @return projected tensor plus bias
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
