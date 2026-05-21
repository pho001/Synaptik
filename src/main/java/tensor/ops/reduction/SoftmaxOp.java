package tensor.ops.reduction;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;

/**
 * Graph-building definition for {@code softmax}.
 */
public final class SoftmaxOp {
    private SoftmaxOp() {
    }

    public static Tensor build(Tensor input, int dimension) {
        ReductionShapeRules.requireFloatingInput(input, "softmax");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor max = input.max(normalizedDimension, true);
        Tensor shifted = input.sub(max);
        Tensor exp = shifted.exp();
        Tensor denominator = exp.sum(normalizedDimension, true);
        Tensor out = exp.div(denominator);
        out.setLabel("softmax");
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor dot = outGrad.mul(out).sum(normalizedDimension, true);
            Tensor grad = out.mul(outGrad.sub(dot));
            context.accumulate(input, grad);
        });
        return out;
    }
}
