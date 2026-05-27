package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
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
            CpuKernelContext context,
            int dimension
    ) {
        if (output.getDataType() != DataType.BOOL) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support " + output.getDataType());
        }
        BoolReduceLoops.execute(input, output, dimension, isAll());
    }
}
