package tensor.ops.unary;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

final class UnarySupport {
    private UnarySupport() {
    }

    static void requireNumeric(Tensor input, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException(opName + " requires numeric input.");
        }
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            TensorInternalAccess.setGradient(input, gradientDelta);
        } else {
            TensorInternalAccess.setGradient(input, input.getGradient().add(gradientDelta));
        }
    }
}
