package backend.cuda.prepare;

import backend.ComputeBackend;
import backend.accelerator.prepare.GpuAcceleratorPrepareSupport;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.cpu.prepare.CpuNodePreparer;
import backend.cuda.bridge.CudaFfmBridge;
import backend.cuda.bridge.CudaGraphBridge;
import backend.cuda.exec.PreparedCudaExecutable;
import backend.cuda.lowering.CudaGpuPartitionPlan;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.optimizer.partition.PartitionPlan;

public final class CudaGpuNodePreparer {
    private final CpuNodePreparer cpuPreparer;
    private final CudaGraphBridge bridge;

    public CudaGpuNodePreparer(CpuNodePreparer cpuPreparer) {
        this(cpuPreparer, new CudaFfmBridge());
    }

    public CudaGpuNodePreparer(CpuNodePreparer cpuPreparer, CudaGraphBridge bridge) {
        this.cpuPreparer = cpuPreparer;
        this.bridge = bridge;
    }

    public CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        PartitionExecutionRole role = context.partitionRoleFor(node.id());
        if (role == PartitionExecutionRole.INTERIOR) {
            return GpuAcceleratorPrepareSupport.interiorMetadata(ComputeBackend.GPU_CUDA, role);
        }
        if (role != PartitionExecutionRole.ANCHOR) {
            return cpuPreparer.prepareAsCpu(node, context);
        }
        LoweredRegion loweredRegion = GpuAcceleratorPrepareSupport.requireLoweredRegion(
                context.cudaLoweredRegionForAnchor(node.id()),
                "CUDA GPU",
                node.id()
        );
        LoweringFamily loweringFamily = GpuAcceleratorPrepareSupport.resolveLoweringFamily(
                loweredRegion,
                LoweringFamily.CUDA_GRAPH_REGION
        );
        PartitionPlan genericPlan = context.backendPlanForAnchor(node.id());
        CudaGpuPartitionPlan plan = GpuAcceleratorPrepareSupport.requirePlan(
                genericPlan,
                CudaGpuPartitionPlan.class,
                "CUDA GPU",
                node.id()
        );
        var fallback = GpuAcceleratorPrepareSupport.prepareCpuFallback(
                plan,
                context,
                cpuPreparer,
                "CUDA GPU",
                false
        );

        return new CompiledNodeExecutionMetadata(
                ComputeBackend.GPU_CUDA,
                null,
                fallback.anchorCpuMetadata().cpuPlan(),
                null,
                null,
                new PreparedCudaExecutable(plan.dagSpec(), loweringFamily, bridge, fallback.preparedSteps()),
                PartitionExecutionRole.ANCHOR
        );
    }
}
