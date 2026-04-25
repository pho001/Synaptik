package backend.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

import java.util.Objects;

public final class BackendPrepareDispatcher {
    private final CpuNodePreparer cpuPreparer;
    private final AppleGpuNodePreparer appleGpuPreparer;
    private final CudaGpuNodePreparer cudaGpuPreparer;

    private BackendPrepareDispatcher(RuntimeConfig runtimeConfig) {
        this.cpuPreparer = new CpuNodePreparer(runtimeConfig);
        this.appleGpuPreparer = new AppleGpuNodePreparer(cpuPreparer);
        this.cudaGpuPreparer = new CudaGpuNodePreparer(cpuPreparer);
    }

    public static BackendPrepareDispatcher from(RuntimeConfig runtimeConfig) {
        return new BackendPrepareDispatcher(Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null"));
    }

    public CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return switch (node.backend()) {
            case CPU -> cpuPreparer.prepare(node, context);
            case GPU_METAL -> appleGpuPreparer.prepare(node, context);
            case GPU_CUDA -> cudaGpuPreparer.prepare(node, context);
            case GPU_OPENCL ->
                    new CompiledNodeExecutionMetadata(node.backend(), null, null, null, null, null, PartitionExecutionRole.NONE);
        };
    }
}
