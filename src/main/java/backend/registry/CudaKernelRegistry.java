package backend.registry;

import backend.kernels.cuda.CudaKernel;
import backend.kernels.cuda.CudaNoopKernel;
import operations.Operation;

import java.util.EnumMap;
import java.util.Map;

public final class CudaKernelRegistry {
    private static final Map<Operation.OpType, CudaKernel> KERNELS = new EnumMap<>(Operation.OpType.class);

    static {
        KERNELS.put(Operation.OpType.NOOP, new CudaNoopKernel());
    }

    private CudaKernelRegistry() {}

    public static CudaKernel resolve(Operation.OpType type) {
        return KERNELS.get(type);
    }
}
