package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

abstract class StorageAwareSumLikeReductionKernel extends StorageAwareReductionKernel {
    protected abstract SumLikeReduction reduction();

    @Override
    protected final Operation.OpType opType() {
        return switch (reduction()) {
            case SUM -> Operation.OpType.SUM;
            case MEAN -> Operation.OpType.MEAN;
        };
    }

    @Override
    protected final void executeArray(
            Operation operation,
            Tensor input,
            Tensor output,
            CpuKernelContext context,
            int dimension
    ) {
        switch (output.getDataType()) {
            case FLOAT64 -> executeF64(input, output, dimension, context);
            case FLOAT32 -> executeF32(input, output, dimension, context);
            case BFLOAT16 -> executeBF16(input, output, dimension, context);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + output.getDataType()
            );
        }
    }

    private void executeF64(Tensor input, Tensor output, int dimension, CpuKernelContext context) {
        SumLoops.execute(input, output, dimension, context);
        reduction().finalizeF64(output, input, dimension);
    }

    private void executeF32(Tensor input, Tensor output, int dimension, CpuKernelContext context) {
        SumLoops.executeF32(input, output, dimension, context);
        reduction().finalizeF32(output, input, dimension);
    }

    private void executeBF16(Tensor input, Tensor output, int dimension, CpuKernelContext context) {
        float[] continuation = context.inputFloatContinuation(0, input.getFlatDataSize());
        if (continuation != null) {
            SumLoops.executeF32ToBF16(input, continuation, output, dimension, context);
            reduction().finalizeBF16(output, input, dimension);
            return;
        }
        SumLoops.executeBF16(input, output, dimension, context);
        reduction().finalizeBF16(output, input, dimension);
    }
}
