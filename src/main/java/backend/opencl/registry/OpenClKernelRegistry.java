package backend.opencl.registry;

import backend.opencl.kernels.OpenClKernel;
import backend.opencl.kernels.OpenClNoopKernel;
import operations.Operation;

import java.util.EnumMap;
import java.util.Map;

/**
 * Registry of legacy per-node OpenCL kernels.
 */
public final class OpenClKernelRegistry {
    private static final Map<Operation.OpType, OpenClKernel> KERNELS = new EnumMap<>(Operation.OpType.class);

    static {
        KERNELS.put(Operation.OpType.NOOP, new OpenClNoopKernel());
    }

    private OpenClKernelRegistry() {}

    /**
     * Returns the registered OpenCL kernel for an operation type, or {@code null}.
     */
    public static OpenClKernel resolve(Operation.OpType type) {
        return KERNELS.get(type);
    }
}
