package backend.kernels.cpu.nn;

import operations.nn.pool.avgPool2d;
import operations.nn.pool.avgPool2dBackwardInput;
import operations.nn.pool.maxPool2d;
import operations.nn.pool.maxPool2dBackwardInput;
import tensor.Tensor;

final class Pool2dExecutor {
    private Pool2dExecutor() {
    }

    static void maxForwardF64(maxPool2d op, Tensor input, Tensor out, int[] argmaxWorkspace) {
        Pool2dDirectBackend.maxForwardF64(op, input, out, argmaxWorkspace);
    }

    static void maxForwardF32(maxPool2d op, Tensor input, Tensor out, int[] argmaxWorkspace) {
        Pool2dDirectBackend.maxForwardF32(op, input, out, argmaxWorkspace);
    }

    static void maxForwardBF16(maxPool2d op, Tensor input, Tensor out, int[] argmaxWorkspace) {
        Pool2dDirectBackend.maxForwardBF16(op, input, out, argmaxWorkspace);
    }

    static void maxBackwardInputF64(maxPool2dBackwardInput op, Tensor outGrad, Tensor gradInput, int[] argmaxWorkspace) {
        Pool2dDirectBackend.maxBackwardInputF64(op, outGrad, gradInput, argmaxWorkspace);
    }

    static void maxBackwardInputF32(maxPool2dBackwardInput op, Tensor outGrad, Tensor gradInput, int[] argmaxWorkspace) {
        Pool2dDirectBackend.maxBackwardInputF32(op, outGrad, gradInput, argmaxWorkspace);
    }

    static void maxBackwardInputBF16(maxPool2dBackwardInput op, Tensor outGrad, Tensor gradInput, int[] argmaxWorkspace) {
        Pool2dDirectBackend.maxBackwardInputBF16(op, outGrad, gradInput, argmaxWorkspace);
    }

    static void avgForwardF64(avgPool2d op, Tensor input, Tensor out) {
        Pool2dDirectBackend.avgForwardF64(op, input, out);
    }

    static void avgForwardF32(avgPool2d op, Tensor input, Tensor out) {
        Pool2dDirectBackend.avgForwardF32(op, input, out);
    }

    static void avgForwardBF16(avgPool2d op, Tensor input, Tensor out) {
        Pool2dDirectBackend.avgForwardBF16(op, input, out);
    }

    static void avgBackwardInputF64(avgPool2dBackwardInput op, Tensor outGrad, Tensor gradInput) {
        Pool2dDirectBackend.avgBackwardInputF64(op, outGrad, gradInput);
    }

    static void avgBackwardInputF32(avgPool2dBackwardInput op, Tensor outGrad, Tensor gradInput) {
        Pool2dDirectBackend.avgBackwardInputF32(op, outGrad, gradInput);
    }

    static void avgBackwardInputBF16(avgPool2dBackwardInput op, Tensor outGrad, Tensor gradInput) {
        Pool2dDirectBackend.avgBackwardInputBF16(op, outGrad, gradInput);
    }
}
