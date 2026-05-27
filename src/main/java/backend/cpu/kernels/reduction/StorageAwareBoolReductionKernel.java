package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

abstract class StorageAwareBoolReductionKernel extends StorageAwareReductionKernel {
    protected abstract boolean isAll();

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
        if (inputView.dtype() != DataType.BOOL || outputView.dtype() != DataType.BOOL) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support "
                    + inputView.dtype() + " -> " + outputView.dtype());
        }
        if (input.getFlatDataSize() != inputView.logicalSize()) {
            throw new IllegalArgumentException(getClass().getSimpleName() + " input storage view size does not match input tensor");
        }
        if (output.getFlatDataSize() != outputView.logicalSize()) {
            throw new IllegalArgumentException(getClass().getSimpleName() + " output storage view size does not match output tensor");
        }
        BoolReduceLoops.execute(inputView, outputView, dimension, isAll());
    }
}
