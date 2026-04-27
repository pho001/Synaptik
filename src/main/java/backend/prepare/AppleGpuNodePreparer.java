package backend.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.apple.bridge.AppleMpsFfmBridge;
import backend.apple.bridge.AppleMpsGraphBridge;
import backend.apple.exec.PreparedAppleGpuExecutable;
import backend.apple.lowering.AppleGpuPartitionPlan;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.optimizer.partition.PartitionPlan;

final class AppleGpuNodePreparer {
    private final CpuNodePreparer cpuPreparer;
    private final AppleMpsGraphBridge bridge;

    AppleGpuNodePreparer(CpuNodePreparer cpuPreparer) {
        this(cpuPreparer, new AppleMpsFfmBridge());
    }

    AppleGpuNodePreparer(CpuNodePreparer cpuPreparer, AppleMpsGraphBridge bridge) {
        this.cpuPreparer = cpuPreparer;
        this.bridge = bridge;
    }

    CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        PartitionExecutionRole role = context.partitionRoleFor(node.id());
        if (role == PartitionExecutionRole.INTERIOR) {
            return GpuAcceleratorPrepareSupport.interiorMetadata(ComputeBackend.GPU_METAL, role);
        }
        if (role != PartitionExecutionRole.ANCHOR) {
            return cpuPreparer.prepareAsCpu(node, context);
        }
        LoweredRegion loweredRegion = GpuAcceleratorPrepareSupport.requireLoweredRegion(
                context.appleLoweredRegionForAnchor(node.id()),
                "Apple GPU",
                node.id()
        );
        LoweringFamily loweringFamily = GpuAcceleratorPrepareSupport.resolveLoweringFamily(
                loweredRegion,
                LoweringFamily.APPLE_GRAPH_REGION
        );

        PartitionPlan genericPlan = context.backendPlanForAnchor(node.id());
        AppleGpuPartitionPlan plan = GpuAcceleratorPrepareSupport.requirePlan(
                genericPlan,
                AppleGpuPartitionPlan.class,
                "Apple GPU",
                node.id()
        );
        var fallback = GpuAcceleratorPrepareSupport.prepareCpuFallback(
                plan,
                context,
                cpuPreparer,
                "Apple GPU",
                true
        );

        return new CompiledNodeExecutionMetadata(
                ComputeBackend.GPU_METAL,
                null,
                fallback.anchorCpuMetadata().cpuPlan(),
                null,
                null,
                new PreparedAppleGpuExecutable(
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
