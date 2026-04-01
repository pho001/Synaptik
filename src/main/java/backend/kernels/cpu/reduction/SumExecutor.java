package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import operations.sum;
import tensor.Tensor;

public final class SumExecutor {
    public void execute(sum op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        SumLoops.execute(input, node, op.getDimension(), context);
    }

    public void executeF32(sum op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        SumLoops.executeF32(input, node, op.getDimension(), context);
    }

    public void executeF16(sum op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        SumLoops.executeF16(input, node, op.getDimension(), context);
    }

    private static void validate(sum op, Tensor input, Tensor node, CpuKernelContext context) {
        if (op == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("sum execution arguments cannot be null");
        }
    }
}
