package tensor.ops.binary;

import tensor.Tensor;
import tensor.TensorInternalAccess;

final class BinarySupport {
    private BinarySupport() {
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            TensorInternalAccess.setGradient(input, gradientDelta);
        } else {
            TensorInternalAccess.setGradient(input, input.getGradient().add(gradientDelta));
        }
    }
}
