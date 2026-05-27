package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernel;

/**
 * Marker for layout kernels that establish output storage identity during execution.
 */
public interface CpuLayoutOutputStorageDeferredKernel extends CpuKernel {
}
