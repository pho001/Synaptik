package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import operations.logSoftmax;
import tensor.Tensor;

public final class LogSoftmaxExecutor {
    public void execute(logSoftmax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        LogSoftmaxLoops.execute(input, node, op.getDimension(), context);
    }

    public void executeF32(logSoftmax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        LogSoftmaxLoops.executeF32(input, node, op.getDimension(), context);
    }

    public void executeF16(logSoftmax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        LogSoftmaxLoops.executeF16(input, node, op.getDimension(), context);
    }

    private static void validate(logSoftmax op, Tensor input, Tensor node, CpuKernelContext context) {
        if (op == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("logSoftmax execution arguments cannot be null");
        }
    }
}
