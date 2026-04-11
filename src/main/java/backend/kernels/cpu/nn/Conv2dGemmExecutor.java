package backend.kernels.cpu.nn;

import backend.kernels.cpu.CpuKernelContext;
import operations.conv2dGemm;
import tensor.Tensor;

final class Conv2dGemmExecutor {
    private Conv2dGemmExecutor() {
    }

    static void forwardF64(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        Conv2dGemmBackend.forwardF64(op, input, weight, bias, out);
    }

    static void forwardF32(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        Conv2dGemmBackend.forwardF32(op, input, weight, bias, out);
    }

    static void forwardBF16(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        Conv2dGemmBackend.forwardBF16(op, input, weight, bias, out, context);
    }
}
