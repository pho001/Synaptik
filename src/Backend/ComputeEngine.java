package Backend;

import Tensor.Tensor;

import java.util.Map;

public class ComputeEngine {

    private static final Map<ComputeBackend, BackendExecutor> executors;

    static {
        executors = Map.of(
                ComputeBackend.CPU, new CPUBackend(),
                ComputeBackend.GPU_CUDA, new CudaBackend(),
                ComputeBackend.GPU_OPENCL, new OpenClBackend()
        );
    }

    public static void compute(Tensor tensor) {
        ComputeBackend backend = tensor.getEffectiveBackend();
        BackendExecutor executor = executors.get(backend);

        if (executor == null) {
            throw new UnsupportedOperationException("Backend " + backend + " is not available");
        }

        executor.execute(tensor.getOperation(), tensor.getPrevTensors(), tensor);
    }
    public static void backward(Tensor tensor) {
        ComputeBackend backend = tensor.getEffectiveBackend();
        BackendExecutor executor = executors.get(backend);

        if (executor == null) {
            throw new UnsupportedOperationException("Backend " + backend + " is not available");
        }

        executor.backward(tensor.getOperation(), tensor.getPrevTensors(), tensor);
    }


}