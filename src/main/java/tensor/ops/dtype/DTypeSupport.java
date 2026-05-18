package tensor.ops.dtype;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

final class DTypeSupport {
    private DTypeSupport() {
    }

    static boolean isFloating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            TensorInternalAccess.setGradient(input, gradientDelta);
        } else {
            TensorInternalAccess.setGradient(input, input.getGradient().add(gradientDelta));
        }
    }
}
