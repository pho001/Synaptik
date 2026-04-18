package backend.kernels.cpu.nn;

import operations.conv2d;
import operations.conv2dBackwardInput;
import operations.conv2dBackwardWeight;
import tensor.Tensor;

final class Conv2dExecutor {
    private Conv2dExecutor() {
    }

    static void forwardF64(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        Conv2dDirectBackend.forwardF64(op, input, weight, bias, out);
    }

    static void forwardF32(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        Conv2dDirectBackend.forwardF32(op, input, weight, bias, out);
    }

    static void forwardBF16(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        Conv2dDirectBackend.forwardBF16(op, input, weight, bias, out);
    }

    static void backwardInputF64(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput) {
        Conv2dDirectBackend.backwardInputF64(op, weight, outGrad, gradInput);
    }

    static void backwardInputF32(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput) {
        Conv2dDirectBackend.backwardInputF32(op, weight, outGrad, gradInput);
    }

    static void backwardInputF16(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput) {
        Conv2dDirectBackend.backwardInputF16(op, weight, outGrad, gradInput);
    }

    static void backwardWeightF64(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight) {
        Conv2dDirectBackend.backwardWeightF64(op, input, outGrad, gradWeight);
    }

    static void backwardWeightF32(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight) {
        Conv2dDirectBackend.backwardWeightF32(op, input, outGrad, gradWeight);
    }

    static void backwardWeightF16(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight) {
        Conv2dDirectBackend.backwardWeightF16(op, input, outGrad, gradWeight);
    }
}
