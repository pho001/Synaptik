package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.CpuKernelContext;
import operations.reduceAll;
import operations.reduceAny;
import tensor.Tensor;

public final class BoolReduceExecutor {
    public void execute(reduceAll op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        BoolReduceLoops.execute(input, node, op.getDimension(), true);
    }

    public void execute(reduceAny op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        BoolReduceLoops.execute(input, node, op.getDimension(), false);
    }

    private static void validate(Object op, Tensor input, Tensor node, CpuKernelContext context) {
        if (op == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("bool reduction execution arguments cannot be null");
        }
    }
}
