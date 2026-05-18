package tensor.ops.linalg;

import graph.optimizer.intent.BackendIntentPropagator;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for matrix multiplication.
 *
 * <p>Shape/dtype validation and backward wiring live here; backend execution
 * continues to consume the immutable {@code operations.linalg.matmul}
 * descriptor produced by {@link MatMulSpec}.</p>
 */
public final class MatMulOp {
    private MatMulOp() {
    }

    /**
     * Multiplies matrices or batches of matrices.
     *
     * @param first left operand; must be non-null and floating numeric
     * @param second right operand; must be non-null and floating numeric
     * @return matrix product tensor with shape inferred from the operands
     */
    public static Tensor build(Tensor first, Tensor second) {
        MatMulSpec spec = MatMulSpec.resolve(first, second);
        Operation op = spec.operation();
        Tensor out = TensorPrimitiveBuilder.binary(first, second, spec.outShape(), op, "matmul", spec.outputType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (first.getRequiresGrad()) {
                Tensor gradRaw = outGrad.matmul(LinalgSupport.transposeLastTwoAxes(second));
                BackendIntentPropagator.preserve(gradRaw, out);
                Tensor gradForFirst = LinalgSupport.sumToShape(gradRaw, first.getShapeUnsafe());
                LinalgSupport.accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = LinalgSupport.transposeLastTwoAxes(first).matmul(outGrad);
                BackendIntentPropagator.preserve(gradRaw, out);
                Tensor gradForSecond = LinalgSupport.sumToShape(gradRaw, second.getShapeUnsafe());
                LinalgSupport.accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }
}
