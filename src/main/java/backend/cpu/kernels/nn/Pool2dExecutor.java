package backend.cpu.kernels.nn;

import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
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

    static void avgForwardF64(avgPool2d op, Tensor input, Tensor out) {
        Pool2dDirectBackend.avgForwardF64(op, input, out);
    }

    static void avgForwardF32(avgPool2d op, Tensor input, Tensor out) {
        Pool2dDirectBackend.avgForwardF32(op, input, out);
    }

    static void avgForwardBF16(avgPool2d op, Tensor input, Tensor out) {
        Pool2dDirectBackend.avgForwardBF16(op, input, out);
    }

}
