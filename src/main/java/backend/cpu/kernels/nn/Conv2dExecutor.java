package backend.cpu.kernels.nn;

import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dBackwardInput;
import operations.nn.conv.conv2dBackwardWeight;
import backend.cpu.kernels.CpuKernelContext;
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

    static void backwardInputF64(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        Conv2dDirectBackend.backwardInputF64(op, weight, outGrad, gradInput, context);
    }

    static void backwardInputF32(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        Conv2dDirectBackend.backwardInputF32(op, weight, outGrad, gradInput, context);
    }

    static void backwardInputF16(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        Conv2dDirectBackend.backwardInputF16(op, weight, outGrad, gradInput, context);
    }

    static void backwardWeightF64(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        Conv2dDirectBackend.backwardWeightF64(op, input, outGrad, gradWeight, context);
    }

    static void backwardWeightF32(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        Conv2dDirectBackend.backwardWeightF32(op, input, outGrad, gradWeight, context);
    }

    static void backwardWeightF16(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        Conv2dDirectBackend.backwardWeightF16(op, input, outGrad, gradWeight, context);
    }
}
