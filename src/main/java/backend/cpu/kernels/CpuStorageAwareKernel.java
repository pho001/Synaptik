package backend.cpu.kernels;

/**
 * Marker for kernels that consume CpuStorageView bindings directly, including strided layouts.
 */
public interface CpuStorageAwareKernel extends CpuKernel {
}
