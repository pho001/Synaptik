package tensor.ops.reduction;

import graph.optimizer.intent.BackendIntentPropagator;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;

/**
 * Graph-building definition for {@code logSoftmax}.
 */
public final class LogSoftmaxOp {
    private LogSoftmaxOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "logSoftmax");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor max = input.max(normalizedDimension, true);
        Tensor shifted = input.sub(max);
        Tensor exp = shifted.exp();
        Tensor denominator = exp.sum(normalizedDimension, true);
        Tensor out = shifted.sub(denominator.log());
        out.setLabel("logSoftmax");
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor probs = out.exp();
            Tensor sumGrad = outGrad.sum(normalizedDimension, true);
            Tensor grad = outGrad.sub(probs.mul(sumGrad));
            BackendIntentPropagator.preserve(grad, out);
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
