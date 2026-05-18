package backend.cpu.kernels.reduction;

import backend.cpu.kernels.*;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.nativecpu.NativeCpuReductionExecutor;
import operations.Operation;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import tensor.Tensor;

public final class MinMaxReduceExecutor {
    public void execute(reduceMin op, Tensor input, Tensor node, CpuKernelContext context) {
        if (NativeCpuReductionExecutor.tryRunMinMax(Operation.OpType.REDUCE_MIN, input, node, op.getDimension(), context)) {
            return;
        }
        validate(op, input, node, context);
        MinMaxReduceLoops.execute(input, node, op.getDimension(), context, false);
    }

    public void execute(reduceMax op, Tensor input, Tensor node, CpuKernelContext context) {
        if (NativeCpuReductionExecutor.tryRunMinMax(Operation.OpType.REDUCE_MAX, input, node, op.getDimension(), context)) {
            return;
        }
        validate(op, input, node, context);
        MinMaxReduceLoops.execute(input, node, op.getDimension(), context, true);
    }

    public void executeF32(reduceMin op, Tensor input, Tensor node, CpuKernelContext context) {
        if (NativeCpuReductionExecutor.tryRunMinMax(Operation.OpType.REDUCE_MIN, input, node, op.getDimension(), context)) {
            return;
        }
        validate(op, input, node, context);
        MinMaxReduceLoops.executeF32(input, node, op.getDimension(), context, false);
    }

    public void executeF32(reduceMax op, Tensor input, Tensor node, CpuKernelContext context) {
        if (NativeCpuReductionExecutor.tryRunMinMax(Operation.OpType.REDUCE_MAX, input, node, op.getDimension(), context)) {
            return;
        }
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
