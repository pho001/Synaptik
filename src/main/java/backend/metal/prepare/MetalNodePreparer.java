package backend.metal.prepare;

import backend.ComputeBackend;
import backend.accelerator.prepare.GpuAcceleratorPrepareSupport;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.cpu.prepare.CpuNodePreparer;
import backend.metal.bridge.MetalMpsFfmBridge;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.metal.exec.PreparedMetalExecutable;
import backend.metal.lowering.MetalPartitionPlan;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.optimizer.partition.PartitionPlan;

public final class MetalNodePreparer {
    private final CpuNodePreparer cpuPreparer;
    private final MetalMpsGraphBridge bridge;

    public MetalNodePreparer(CpuNodePreparer cpuPreparer) {
        this(cpuPreparer, new MetalMpsFfmBridge());
    }

    public MetalNodePreparer(CpuNodePreparer cpuPreparer, MetalMpsGraphBridge bridge) {
        this.cpuPreparer = cpuPreparer;
        this.bridge = bridge;
    }

    public CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        PartitionExecutionRole role = context.partitionRoleFor(node.id());
        if (role == PartitionExecutionRole.INTERIOR) {
            return GpuAcceleratorPrepareSupport.interiorMetadata(ComputeBackend.GPU_METAL, role);
        }
        if (role != PartitionExecutionRole.ANCHOR) {
            return cpuPreparer.prepareAsCpu(node, context);
        }
        LoweredRegion loweredRegion = GpuAcceleratorPrepareSupport.requireLoweredRegion(
                context.metalLoweredRegionForAnchor(node.id()),
                "Metal GPU",
                node.id()
        );
        LoweringFamily loweringFamily = GpuAcceleratorPrepareSupport.resolveLoweringFamily(
                loweredRegion,
                LoweringFamily.METAL_GRAPH_REGION
        );

        PartitionPlan genericPlan = context.backendPlanForAnchor(node.id());
        MetalPartitionPlan plan = GpuAcceleratorPrepareSupport.requirePlan(
                genericPlan,
                MetalPartitionPlan.class,
                "Metal GPU",
                node.id()
        );
        var fallback = GpuAcceleratorPrepareSupport.prepareCpuFallback(
                plan,
                context,
                cpuPreparer,
                "Metal GPU",
                true
        );

        return new CompiledNodeExecutionMetadata(
                ComputeBackend.GPU_METAL,
                null,
                fallback.anchorCpuMetadata().cpuPlan(),
                null,
                null,
                new PreparedMetalExecutable(
                        plan,
                        loweringFamily,
                        fallback.computeNode(),
                        fallback.computeCpuMetadata().cpuPlan(),
                        bridge,
                        fallback.preparedSteps()
                ),
                PartitionExecutionRole.ANCHOR
        );
    }
}
