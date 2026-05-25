package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import tensor.Tensor;

public final class BoolReduceExecutor {
    public void execute(reduceAll op, Tensor input, Tensor node, CpuKernelContext context) {
        if (ReductionStorageLoops.tryRunBool(Operation.OpType.REDUCE_ALL, input, node, op.getDimension(), context)) {
            return;
        }
        validate(op, input, node, context);
        BoolReduceLoops.execute(input, node, op.getDimension(), true);
    }

    public void execute(reduceAny op, Tensor input, Tensor node, CpuKernelContext context) {
        if (ReductionStorageLoops.tryRunBool(Operation.OpType.REDUCE_ANY, input, node, op.getDimension(), context)) {
            return;
        }
        validate(op, input, node, context);
        BoolReduceLoops.execute(input, node, op.getDimension(), false);
    }

    private static void validate(Object op, Tensor input, Tensor node, CpuKernelContext context) {
        if (op == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("bool reduction execution arguments cannot be null");
        }
    }
}
