package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.CpuKernelContext;
import operations.reduceMax;
import operations.reduceMin;
import tensor.Tensor;

public final class MinMaxReduceExecutor {
    public void execute(reduceMin op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        MinMaxReduceLoops.execute(input, node, op.getDimension(), context, false);
    }

    public void execute(reduceMax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        MinMaxReduceLoops.execute(input, node, op.getDimension(), context, true);
    }

    public void executeF32(reduceMin op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        MinMaxReduceLoops.executeF32(input, node, op.getDimension(), context, false);
    }

    public void executeF32(reduceMax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        MinMaxReduceLoops.executeF32(input, node, op.getDimension(), context, true);
    }

    public void executeBF16(reduceMin op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        MinMaxReduceLoops.executeBF16(input, node, op.getDimension(), context, false);
    }

    public void executeBF16(reduceMax op, Tensor input, Tensor node, CpuKernelContext context) {
        validate(op, input, node, context);
        MinMaxReduceLoops.executeBF16(input, node, op.getDimension(), context, true);
    }

    private static void validate(Object op, Tensor input, Tensor node, CpuKernelContext context) {
        if (op == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("min/max reduction execution arguments cannot be null");
        }
    }
}
