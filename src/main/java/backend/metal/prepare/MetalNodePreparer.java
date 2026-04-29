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

/**
 * Prepares compiled nodes for Metal partition execution.
 *
 * <p>Anchor nodes receive a {@link PreparedMetalExecutable}; interior nodes are
 * marked as covered by the partition, and non-partition nodes fall back to CPU
 * preparation.</p>
 */
public final class MetalNodePreparer {
    private final CpuNodePreparer cpuPreparer;
    private final MetalMpsGraphBridge bridge;

    /**
     * Creates a preparer using the default FFM Metal bridge.
     */
    public MetalNodePreparer(CpuNodePreparer cpuPreparer) {
        this(cpuPreparer, new MetalMpsFfmBridge());
    }

    /**
     * Creates a preparer with an explicit Metal bridge implementation.
     */
    public MetalNodePreparer(CpuNodePreparer cpuPreparer, MetalMpsGraphBridge bridge) {
        this.cpuPreparer = cpuPreparer;
        this.bridge = bridge;
    }

    /**
     * Prepares execution metadata for a node according to its Metal partition role.
     */
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
                        bridge,
                        fallback.preparedSteps(),
                        context.runtimeConfig().accelerator().metal()
                ),
                PartitionExecutionRole.ANCHOR
        );
    }
}
