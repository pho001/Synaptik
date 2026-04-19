package tensor.ops.linalg;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

public final class TensorLinearOps {
    private TensorLinearOps() {
    }

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
                LinalgSupport.accumulateGradient(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = LinalgSupport.transposeLastTwoAxes(input).matmul(outGrad);
                LinalgSupport.accumulateGradient(weight, LinalgSupport.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
        });
        return out;
    }

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
                LinalgSupport.accumulateGradient(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = LinalgSupport.transposeLastTwoAxes(input).matmul(outGrad);
                LinalgSupport.accumulateGradient(weight, LinalgSupport.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
            if (bias.getRequiresGrad()) {
                LinalgSupport.accumulateGradient(bias, LinalgSupport.sumToShape(outGrad, bias.getShapeUnsafe()));
            }
        });
        return out;
    }
}
