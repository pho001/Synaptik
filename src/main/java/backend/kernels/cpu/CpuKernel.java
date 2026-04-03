package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public interface CpuKernel {
    default void forwardF64(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT64");
    }

    default void forwardF32(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT32");
    }

    default void forwardF16(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT16");
    }

    default void forwardBOOL(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support BOOL");
    }

    default CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.MEDIUM;
    }
}
