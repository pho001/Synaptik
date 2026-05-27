package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import tensor.Tensor;

abstract class StorageAwareMinMaxReductionKernel extends StorageAwareReductionKernel {
    protected abstract boolean isMax();

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
            case FLOAT64 -> MinMaxReduceLoops.execute(inputView, outputView, dimension, context, isMax());
            case FLOAT32 -> MinMaxReduceLoops.executeF32(inputView, outputView, dimension, context, isMax());
            case BFLOAT16 -> MinMaxReduceLoops.executeBF16(inputView, outputView, dimension, context, isMax());
            case INT32 -> MinMaxReduceLoops.executeI32(inputView, outputView, dimension, context, isMax());
            case INT64 -> MinMaxReduceLoops.executeI64(inputView, outputView, dimension, context, isMax());
            case BOOL -> MinMaxReduceLoops.executeBOOL(inputView, outputView, dimension, context, isMax());
        }
    }
}
