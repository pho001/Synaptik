package backend.registry;

import backend.kernels.opencl.OpenClKernel;
import backend.kernels.opencl.OpenClNoopKernel;
import operations.Operation;

import java.util.EnumMap;
import java.util.Map;

public final class OpenClKernelRegistry {
    private static final Map<Operation.OpType, OpenClKernel> KERNELS = new EnumMap<>(Operation.OpType.class);

    static {
        KERNELS.put(Operation.OpType.NOOP, new OpenClNoopKernel());
    }

    private OpenClKernelRegistry() {}

    public static OpenClKernel resolve(Operation.OpType type) {
        return KERNELS.get(type);
    }
}
