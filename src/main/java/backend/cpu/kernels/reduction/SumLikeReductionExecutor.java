package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.nativecpu.NativeCpuReductionExecutor;
import operations.Operation;
import tensor.Tensor;

final class SumLikeReductionExecutor {
    private SumLikeReductionExecutor() {}

    static void executeF64(SumLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        if (NativeCpuReductionExecutor.tryRunSumLike(opType(reduction), input, node, dimension, context)) {
            return;
        }
        SumLoops.execute(input, node, dimension, context);
        reduction.finalizeF64(node, input, dimension);
    }

    static void executeF32(SumLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        if (NativeCpuReductionExecutor.tryRunSumLike(opType(reduction), input, node, dimension, context)) {
            return;
        }
        SumLoops.executeF32(input, node, dimension, context);
        reduction.finalizeF32(node, input, dimension);
    }

    static void executeBF16(SumLikeReduction reduction, Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validate(reduction, input, node, context);
        float[] continuation = context.inputFloatContinuation(0, input.getFlatDataSize());
        if (continuation != null) {
            SumLoops.executeF32ToBF16(input, continuation, node, dimension, context);
            reduction.finalizeBF16(node, input, dimension);
            return;
        }
        SumLoops.executeBF16(input, node, dimension, context);
        reduction.finalizeBF16(node, input, dimension);
    }

    private static void validate(SumLikeReduction reduction, Tensor input, Tensor node, CpuKernelContext context) {
        if (reduction == null || input == null || node == null || context == null) {
            throw new IllegalArgumentException("sum-like reduction execution arguments cannot be null");
        }
    }

    private static Operation.OpType opType(SumLikeReduction reduction) {
        return switch (reduction) {
            case SUM -> Operation.OpType.SUM;
            case MEAN -> Operation.OpType.MEAN;
        };
    }
}
