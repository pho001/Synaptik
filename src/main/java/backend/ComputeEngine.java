package backend;

import backend.accelerator.exec.PartitionExecutionRole;
import backend.apple.AppleGpuBackend;
import backend.cuda.CudaGpuBackend;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

public final class ComputeEngine {
    private static final CPUBackend CPU_BACKEND = new CPUBackend();
    private static final CudaBackend CUDA_BACKEND = new CudaBackend();
    private static final CudaGpuBackend CUDA_GPU_BACKEND = new CudaGpuBackend();
    private static final OpenClBackend OPENCL_BACKEND = new OpenClBackend();
    private static final AppleGpuBackend APPLE_GPU_BACKEND = new AppleGpuBackend();

    private ComputeEngine() {}

    public static void compute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        if (node == null) {
            throw new IllegalArgumentException("node cannot be null");
        }
        if (metadata == null) {
            throw new IllegalArgumentException("metadata cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        if (metadata.partitionRole() == PartitionExecutionRole.INTERIOR) {
            return;
        }
        switch (metadata.backend()) {
            case CPU -> CPU_BACKEND.execute(node, metadata, context);
            case GPU_CUDA -> {
                if (metadata.acceleratorExecutable() != null) {
                    CUDA_GPU_BACKEND.execute(node, metadata, context);
                } else {
                    CUDA_BACKEND.execute(node, metadata, context);
                }
            }
            case GPU_OPENCL -> OPENCL_BACKEND.execute(node, metadata, context);
            case GPU_METAL -> APPLE_GPU_BACKEND.execute(node, metadata, context);
            default -> throw new UnsupportedOperationException(
                    "Backend " + metadata.backend() + " is not available"
            );
        }
    }
}
