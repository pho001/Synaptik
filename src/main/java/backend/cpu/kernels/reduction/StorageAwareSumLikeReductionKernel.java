package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.storage.CpuStorageView;
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
            CpuStorageView inputView,
            CpuStorageView outputView,
            CpuKernelContext context,
            int dimension
    ) {
        if (inputView.dtype() != outputView.dtype()) {
            throw new IllegalArgumentException(getClass().getSimpleName() + " requires input and output dtypes to match");
        }
        if (input.getFlatDataSize() != inputView.logicalSize()) {
            throw new IllegalArgumentException(getClass().getSimpleName() + " input storage view size does not match input tensor");
        }
        if (output.getFlatDataSize() != outputView.logicalSize()) {
            throw new IllegalArgumentException(getClass().getSimpleName() + " output storage view size does not match output tensor");
        }
        switch (outputView.dtype()) {
            case FLOAT64 -> executeF64(inputView, outputView, dimension, context);
            case FLOAT32 -> executeF32(inputView, outputView, dimension, context);
            case BFLOAT16 -> executeBF16(inputView, outputView, dimension, context);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + outputView.dtype()
            );
        }
    }

    private void executeF64(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        SumLoops.execute(input, output, dimension, context);
        reduction().finalizeF64(output, input, dimension);
    }

    private void executeF32(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        SumLoops.executeF32(input, output, dimension, context);
        reduction().finalizeF32(output, input, dimension);
    }

    private void executeBF16(CpuStorageView input, CpuStorageView output, int dimension, CpuKernelContext context) {
        float[] continuation = context.inputFloatContinuation(0, input.logicalSize());
        if (continuation != null) {
            SumLoops.executeF32ToBF16(input, continuation, output, dimension, context);
            reduction().finalizeBF16(output, input, dimension);
            return;
        }
        SumLoops.executeBF16(input, output, dimension, context);
        reduction().finalizeBF16(output, input, dimension);
    }
}
