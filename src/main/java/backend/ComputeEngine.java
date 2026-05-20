package backend;

import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.cpu.CpuBackend;
import backend.cuda.CudaBackend;
import backend.cuda.CudaGpuBackend;
import backend.metal.MetalBackend;
import backend.opencl.OpenClBackend;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;

/**
 * Runtime dispatcher from prepared node metadata to the concrete backend implementation.
 *
 * <p>Prepared execution calls this class for each executable step. The method inspects
 * {@link CompiledNodeExecutionMetadata#backend()} and delegates to CPU, CUDA, OpenCL, or Metal backend
 * implementations. Nodes marked as partition interiors are skipped because their work is owned by the
 * partition anchor step.</p>
 *
 * <p>The engine stores backend singletons and has no per-run mutable state. Per-run state is carried by
 * {@link ExecutionContext}.</p>
 */
public final class ComputeEngine {
    private static final CpuBackend CPU_BACKEND = new CpuBackend();
    private static final CudaBackend CUDA_BACKEND = new CudaBackend();
    private static final CudaGpuBackend CUDA_GPU_BACKEND = new CudaGpuBackend();
    private static final OpenClBackend OPENCL_BACKEND = new OpenClBackend();
    private static final MetalBackend METAL_BACKEND = new MetalBackend();

    private ComputeEngine() {}

    /**
     * Executes one prepared graph node or partition anchor.
     *
     * @param node compiled node to execute; must not be {@code null}
     * @param metadata prepare-time execution metadata; must not be {@code null}
     * @param context per-run execution context and runtime state; must not be {@code null}
     * @throws IllegalArgumentException if any argument is {@code null}
     * @throws UnsupportedOperationException if the selected backend has no runtime implementation
     */
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
                if (metadata.artifact() instanceof AcceleratorExecutionArtifact) {
                    CUDA_GPU_BACKEND.execute(node, metadata, context);
                } else {
                    CUDA_BACKEND.execute(node, metadata, context);
                }
            }
            case GPU_OPENCL -> OPENCL_BACKEND.execute(node, metadata, context);
            case GPU_METAL -> METAL_BACKEND.execute(node, metadata, context);
            default -> throw new UnsupportedOperationException(
                    "Backend " + metadata.backend() + " is not available"
            );
        }
    }
}
