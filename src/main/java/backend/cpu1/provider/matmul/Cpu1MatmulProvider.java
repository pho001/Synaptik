package backend.cpu1.provider.matmul;

import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import tensor.DataType;

/**
 * Prepare-time matmul provider boundary for cpu1 routes.
 */
public interface Cpu1MatmulProvider {
    Cpu1MatmulRoute route();

    Cpu1MatmulKernelId kernelId(DataType dataType);
}
