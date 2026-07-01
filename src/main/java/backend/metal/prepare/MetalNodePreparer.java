package backend.metal.prepare;

import backend.contract.ComputeBackend;
import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.accelerator.prepare.GpuAcceleratorPrepareSupport;
import backend.cpu.prepare.CpuNodePreparer;
import backend.metal.bridge.MetalMpsFfmCustomKernelBridge;
import backend.metal.bridge.MetalMpsFfmBridge;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.metal.exec.PreparedMetalExecutable;
import backend.metal.kernel.MetalCustomKernelBridge;
import backend.metal.lowering.MetalPartitionPlan;
import backend.lowering.LoweredPartition;
import backend.lowering.LoweringFamily;
import prepare.context.BackendPrepareContext;
import prepare.context.PartitionExecutionRole;
import prepare.validation.BackendPartitionExecutionPlanValidator;
import graph.model.CompiledNode;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.OutputResidencyEffect;
import runtime.execution.InputResidencyRequirement;
import planning.partition.PartitionPlan;

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
    public PreparedStepMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        PartitionExecutionRole role = context.partitionRoleFor(node.id());
        if (role == PartitionExecutionRole.INTERIOR) {
            throw new IllegalStateException("Interior Metal partition node must be covered before prepare: nodeId="
                    + node.id());
        }
        if (role != PartitionExecutionRole.ANCHOR) {
            return cpuPreparer.prepareAsCpu(node, context);
        }
        LoweredPartition loweredPartition = GpuAcceleratorPrepareSupport.requireLoweredPartition(
                context.metalLoweredPartitionForAnchor(node.id()),
                "Metal GPU",
                node.id()
        );
        return preparePartitionStep(loweredPartition, context);
    }

    public PreparedStepMetadata preparePartitionStep(
            LoweredPartition loweredPartition,
            BackendPrepareContext context
    ) {
        LoweringFamily loweringFamily = GpuAcceleratorPrepareSupport.resolveLoweringFamily(
                loweredPartition,
                LoweringFamily.METAL_GRAPH_PARTITION
        );
        var partitionPlan = loweredPartition.units().getFirst().requirePartitionPlan();
        BackendPartitionExecutionPlanValidator.requireBoundaryCoverage(partitionPlan, context);

        PartitionPlan genericPlan = context.backendPlanForAnchor(partitionPlan.anchorNodeId());
        MetalPartitionPlan plan = GpuAcceleratorPrepareSupport.requirePlan(
                genericPlan,
                MetalPartitionPlan.class,
                "Metal GPU",
                partitionPlan.anchorNodeId()
        );
        var fallback = GpuAcceleratorPrepareSupport.prepareCpuFallback(
                plan,
                context,
                cpuPreparer,
                "Metal GPU",
                true
        );

        PreparedMetalExecutable executable = new PreparedMetalExecutable(
                plan,
                loweringFamily,
                partitionPlan,
                bridge,
                fallback.preparedSteps(),
                context.runtimeConfig().accelerator().metal(),
                customKernelBridge
        );
        return new PreparedStepMetadata(
                ComputeBackend.GPU_METAL,
                null,
                java.util.List.of(),
                new AcceleratorExecutionArtifact(executable),
                InputResidencyRequirement.none(),
                OutputResidencyEffect.cpuCurrentIfUnset(executable.outputResidencyReason())
        );
    }
}
