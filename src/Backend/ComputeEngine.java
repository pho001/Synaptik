package Backend;

import Config.backend.CpuKernelConfig;
import Tensor.Tensor;

public class ComputeEngine {
    private static final CPUBackend CPU_BACKEND = new CPUBackend();
    private static final CudaBackend CUDA_BACKEND = new CudaBackend();
    private static final OpenClBackend OPENCL_BACKEND = new OpenClBackend();
    private static volatile ApproxMode approxMode = ApproxMode.OFF;
    private static final ThreadLocal<Boolean> TRAINING_EXECUTION = ThreadLocal.withInitial(() -> Boolean.FALSE);

    public static void compute(Tensor tensor) {
        ComputeBackend backend = tensor.resolveBackend();
        compute(tensor, backend);
    }

    public static void compute(Tensor tensor, ComputeBackend backend) {
        if (backend == null) {
            backend = tensor.resolveBackend();
        }
        switch (backend) {
            case CPU -> CPU_BACKEND.execute(tensor.getOperation(), tensor.getPrevTensors(), tensor);
            case GPU_CUDA -> CUDA_BACKEND.execute(tensor.getOperation(), tensor.getPrevTensors(), tensor);
            case GPU_OPENCL -> OPENCL_BACKEND.execute(tensor.getOperation(), tensor.getPrevTensors(), tensor);
            default -> throw new UnsupportedOperationException("Backend " + backend + " is not available");
        }
    }

    public static void setCpuKernelConfig(CpuKernelConfig config) {
        CPU_BACKEND.setKernelConfig(config);
    }

    public static void setApproxMode(ApproxMode mode) {
        approxMode = mode == null ? ApproxMode.OFF : mode;
    }

    public static ApproxMode getApproxMode() {
        return approxMode;
    }

    public static void setTrainingExecution(boolean trainingExecution) {
        TRAINING_EXECUTION.set(trainingExecution);
    }

    public static void clearTrainingExecution() {
        TRAINING_EXECUTION.remove();
    }

    public static boolean useFastExpApprox() {
        return switch (approxMode) {
            case OFF -> false;
            case ALWAYS -> true;
            case TRAINING_ONLY -> TRAINING_EXECUTION.get();
        };
    }

    public static boolean useFastTanhApprox() {
        return switch (approxMode) {
            case OFF -> false;
            case ALWAYS -> true;
            case TRAINING_ONLY -> TRAINING_EXECUTION.get();
        };
    }
}
