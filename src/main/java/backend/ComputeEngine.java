package backend;

import backend.runtime.ExecutionContext;
import graph.execution.CompiledNodeExecutionMetadata;
import tensor.Tensor;

public final class ComputeEngine {
    private static final CPUBackend CPU_BACKEND = new CPUBackend();
    private static final CudaBackend CUDA_BACKEND = new CudaBackend();
    private static final OpenClBackend OPENCL_BACKEND = new OpenClBackend();

    private ComputeEngine() {}

    public static void compute(
            Tensor node,
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
        switch (metadata.backend()) {
            case CPU -> CPU_BACKEND.execute(node, metadata, context);
            case GPU_CUDA -> CUDA_BACKEND.execute(node, metadata, context);
            case GPU_OPENCL -> OPENCL_BACKEND.execute(node, metadata, context);
            default -> throw new UnsupportedOperationException(
                    "Backend " + metadata.backend() + " is not available"
            );
        }
    }
}
