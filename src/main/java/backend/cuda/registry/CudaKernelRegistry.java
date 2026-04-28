package backend.cuda.registry;

import backend.cuda.kernels.CudaKernel;
import backend.cuda.kernels.CudaNoopKernel;
import operations.Operation;

import java.util.EnumMap;
import java.util.Map;

/**
 * Registry of legacy per-node CUDA kernels.
 */
public final class CudaKernelRegistry {
    private static final Map<Operation.OpType, CudaKernel> KERNELS = new EnumMap<>(Operation.OpType.class);

    static {
        KERNELS.put(Operation.OpType.NOOP, new CudaNoopKernel());
    }

    private CudaKernelRegistry() {}

    /**
     * Returns the registered CUDA kernel for an operation type, or {@code null}.
     */
    public static CudaKernel resolve(Operation.OpType type) {
        return KERNELS.get(type);
    }
}
