package Backend;

import Config.backend.CpuKernelConfig;
import Tensor.Tensor;

public class ComputeEngine {
    private static final CPUBackend CPU_BACKEND = new CPUBackend();
    private static final CudaBackend CUDA_BACKEND = new CudaBackend();
    private static final OpenClBackend OPENCL_BACKEND = new OpenClBackend();

    public static void compute(Tensor tensor) {
        ComputeBackend backend = tensor.resolveBackend();
        compute(tensor, backend);
    }

    public static void compute(Tensor tensor, ComputeBackend backend) {
        if (backend == null) {
            backend = tensor.resolveBackend();
        }
        if (tensor.getPrevTensors() != null) {
            for (Tensor input : tensor.getPrevTensors()) {
                if (input == null) continue;
                // Persist potential user updates and re-materialize dtype-quantized values.
                input.syncDataToStorage();
                input.syncStorageToData();
            }
        }
        switch (backend) {
            case CPU -> CPU_BACKEND.execute(tensor.getOperation(), tensor.getPrevTensors(), tensor);
            case GPU_CUDA -> CUDA_BACKEND.execute(tensor.getOperation(), tensor.getPrevTensors(), tensor);
            case GPU_OPENCL -> OPENCL_BACKEND.execute(tensor.getOperation(), tensor.getPrevTensors(), tensor);
            default -> throw new UnsupportedOperationException("Backend " + backend + " is not available");
        }
        tensor.syncDataToStorage();
        tensor.syncStorageToData();
    }

    public static void setCpuKernelConfig(CpuKernelConfig config) {
        CPU_BACKEND.setKernelConfig(config);
    }
}
