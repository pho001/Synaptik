package backend.metal.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.accelerator.prepare.GpuAcceleratorPrepareSupport;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.cpu.prepare.CpuNodePreparer;
import backend.metal.bridge.MetalMpsFfmCustomKernelBridge;
import backend.metal.bridge.MetalMpsFfmBridge;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.metal.exec.PreparedMetalExecutable;
import backend.metal.kernel.MetalCustomKernelBridge;
import backend.metal.lowering.MetalPartitionPlan;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import backend.prepare.RegionPlanValidator;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.compile.planning.partition.PartitionPlan;

import java.util.List;

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
    private final MetalCustomKernelBridge customKernelBridge;

    /**
     * Creates a preparer using the default FFM Metal bridge.
     */
    public MetalNodePreparer(CpuNodePreparer cpuPreparer) {
        this(cpuPreparer, new MetalMpsFfmBridge(), new MetalMpsFfmCustomKernelBridge());
    }

    /**
     * Creates a preparer with an explicit Metal bridge implementation.
     */
    public MetalNodePreparer(CpuNodePreparer cpuPreparer, MetalMpsGraphBridge bridge) {
        this(cpuPreparer, bridge, MetalCustomKernelBridge.unavailable());
    }

    /**
     * Creates a preparer with explicit Metal graph and custom-kernel bridge implementations.
     */
    public MetalNodePreparer(
            CpuNodePreparer cpuPreparer,
            MetalMpsGraphBridge bridge,
            MetalCustomKernelBridge customKernelBridge
    ) {
        this.cpuPreparer = cpuPreparer;
        this.bridge = bridge;
        this.customKernelBridge = customKernelBridge == null ? MetalCustomKernelBridge.unavailable() : customKernelBridge;
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
        var regionPlan = loweredRegion.units().getFirst().requireRegionPlan();
        RegionPlanValidator.requireBoundaryCoverage(regionPlan, context);

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
                PartitionExecutionRole.ANCHOR,
                null,
                List.of(),
                new AcceleratorExecutionArtifact(new PreparedMetalExecutable(
                        plan,
                        loweringFamily,
                        regionPlan,
                        bridge,
                        fallback.preparedSteps(),
                        context.runtimeConfig().accelerator().metal(),
                        customKernelBridge
                ))
        );
    }
}
