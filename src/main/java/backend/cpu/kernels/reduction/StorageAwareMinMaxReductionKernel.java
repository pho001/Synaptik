package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

abstract class StorageAwareMinMaxReductionKernel extends StorageAwareReductionKernel {
    protected abstract boolean isMax();

    @Override
    protected final void executeArray(
            Operation operation,
            Tensor input,
            Tensor output,
            CpuKernelContext context,
            int dimension
    ) {
        switch (output.getDataType()) {
            case FLOAT64 -> MinMaxReduceLoops.execute(input, output, dimension, context, isMax());
            case FLOAT32 -> MinMaxReduceLoops.executeF32(input, output, dimension, context, isMax());
            case BFLOAT16 -> MinMaxReduceLoops.executeBF16(input, output, dimension, context, isMax());
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + output.getDataType()
            );
        }
    }
}
