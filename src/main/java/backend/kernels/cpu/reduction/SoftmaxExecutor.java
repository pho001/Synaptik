package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.CpuKernelContext;
import operations.softmax;
import tensor.Tensor;

public final class SoftmaxExecutor {
    public void execute(softmax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        SoftmaxLoops.execute(input, node, op.getDimension(), context);
    }

    public void executeF32(softmax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        SoftmaxLoops.executeF32(input, node, op.getDimension(), context);
    }

    public void executeBF16(softmax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        SoftmaxLoops.executeBF16(input, node, op.getDimension(), context);
    }

    private static void validate(softmax op, Tensor input, Tensor node, CpuKernelContext context) {
        if (op == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("softmax execution arguments cannot be null");
        }
    }
}
