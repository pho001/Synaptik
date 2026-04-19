package backend.prepare;

import backend.ComputeBackend;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

import java.util.Objects;

public final class BackendPrepareDispatcher {
    private final CpuNodePreparer cpuPreparer;

    private BackendPrepareDispatcher(RuntimeConfig runtimeConfig) {
        this.cpuPreparer = new CpuNodePreparer(runtimeConfig);
    }

    public static BackendPrepareDispatcher from(RuntimeConfig runtimeConfig) {
        return new BackendPrepareDispatcher(Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null"));
    }

    public CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return switch (node.backend()) {
            case CPU -> cpuPreparer.prepare(node, context);
            case GPU_CUDA, GPU_OPENCL -> new CompiledNodeExecutionMetadata(node.backend(), null, null, null, null);
        };
    }
}
