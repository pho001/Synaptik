package tensor.ops.select;

import tensor.Tensor;

final class SelectSupport {
    private SelectSupport() {
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
