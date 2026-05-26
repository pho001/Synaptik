package backend.cpu.kernels.nn;

import operations.nn.conv.conv2d;
import backend.cpu.execution.CpuKernelContext;
import tensor.Tensor;

final class Conv2dExecutor {
    private Conv2dExecutor() {
    }

    static void forwardF64(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        Conv2dDirectBackend.forwardF64(op, input, weight, bias, out, context);
    }

    static void forwardF32(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        Conv2dDirectBackend.forwardF32(op, input, weight, bias, out, context);
    }

    static void forwardBF16(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        Conv2dDirectBackend.forwardBF16(op, input, weight, bias, out, context);
    }

}
