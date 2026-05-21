package backend.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.cpu.prepare.CpuNodePreparer;
import backend.cuda.prepare.CudaGpuNodePreparer;
import backend.metal.prepare.MetalNodePreparer;
import backend.lowering.LoweredExecutionUnit;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.compile.planning.partition.PartitionPlan;

import java.util.Objects;

public final class BackendPrepareDispatcher {
    private final CpuNodePreparer cpuPreparer;
    private MetalNodePreparer metalPreparer;
    private CudaGpuNodePreparer cudaGpuPreparer;

    private BackendPrepareDispatcher(RuntimeConfig runtimeConfig) {
        this.cpuPreparer = new CpuNodePreparer(runtimeConfig);
    }

    public static BackendPrepareDispatcher from(RuntimeConfig runtimeConfig) {
        return new BackendPrepareDispatcher(Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null"));
    }

    public CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return switch (executionBackendFor(node, context)) {
            case CPU -> cpuPreparer.prepare(node, context);
            case GPU_METAL -> metalPreparer().prepare(node, context);
            case GPU_CUDA -> cudaGpuPreparer().prepare(node, context);
            case GPU_OPENCL ->
                    new CompiledNodeExecutionMetadata(node.backend(), null, java.util.List.of(), null);
        };
    }

    public CompiledNodeExecutionMetadata prepareCpuFusedStep(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(outputNode, "outputNode cannot be null");
        Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return cpuPreparer.prepareLoweredFusedStep(outputNode, loweredUnit, context);
    }

    public CompiledNodeExecutionMetadata prepareCpuNativeRegionStep(
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return cpuPreparer.prepareNativeCpuRegionStep(loweredUnit, context);
    }

    public CompiledNodeExecutionMetadata prepareMetalRegionStep(
            backend.lowering.LoweredRegion loweredRegion,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(loweredRegion, "loweredRegion cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return metalPreparer().prepareRegionStep(loweredRegion, context);
    }

    public CompiledNodeExecutionMetadata prepareCudaRegionStep(
            backend.lowering.LoweredRegion loweredRegion,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(loweredRegion, "loweredRegion cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return cudaGpuPreparer().prepareRegionStep(loweredRegion, context);
    }

    private MetalNodePreparer metalPreparer() {
        if (metalPreparer == null) {
            metalPreparer = new MetalNodePreparer(cpuPreparer);
        }
        return metalPreparer;
    }

    private CudaGpuNodePreparer cudaGpuPreparer() {
        if (cudaGpuPreparer == null) {
            cudaGpuPreparer = new CudaGpuNodePreparer(cpuPreparer);
        }
        return cudaGpuPreparer;
    }

    private ComputeBackend executionBackendFor(CompiledNode node, BackendPrepareContext context) {
        if (context.partitionRoleFor(node.id()) == PartitionExecutionRole.ANCHOR) {
            PartitionPlan selectedPlan = context.backendPlanForAnchor(node.id());
            if (selectedPlan != null && selectedPlan.backend() != null) {
                return selectedPlan.backend();
            }
        }
        return node.backend();
    }
}
