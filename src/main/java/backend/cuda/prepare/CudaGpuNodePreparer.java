package backend.cuda.prepare;

import backend.contract.ComputeBackend;
import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.accelerator.prepare.GpuAcceleratorPrepareSupport;
import backend.cpu.prepare.CpuNodePreparer;
import backend.cuda.bridge.CudaFfmBridge;
import backend.cuda.bridge.CudaGraphBridge;
import backend.cuda.exec.PreparedCudaExecutable;
import backend.cuda.exec.CudaDirectPreparedExecutable;
import backend.cuda.lowering.CudaGpuPartitionPlan;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import prepare.context.BackendPrepareContext;
import prepare.context.PartitionExecutionRole;
import prepare.validation.RegionPlanValidator;
import graph.model.CompiledNode;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.OutputResidencyEffect;
import runtime.execution.InputResidencyRequirement;
import planning.partition.PartitionPlan;

/**
 * Prepares compiled nodes for CUDA partition execution.
 *
 * <p>Anchor nodes receive a {@link PreparedCudaExecutable}; interior nodes are
 * marked as covered by the partition, and non-partition nodes fall back to CPU
 * preparation.</p>
 */
public final class CudaGpuNodePreparer {
    private final CpuNodePreparer cpuPreparer;
    private final CudaGraphBridge bridge;

    /**
     * Creates a preparer using the default FFM CUDA bridge.
     */
    public CudaGpuNodePreparer(CpuNodePreparer cpuPreparer) {
        this(cpuPreparer, new CudaFfmBridge());
    }

    /**
     * Creates a preparer with an explicit CUDA bridge implementation.
     */
    public CudaGpuNodePreparer(CpuNodePreparer cpuPreparer, CudaGraphBridge bridge) {
        this.cpuPreparer = cpuPreparer;
        this.bridge = bridge;
    }

    /**
     * Prepares execution metadata for a node according to its CUDA partition role.
     */
    public PreparedStepMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        PartitionExecutionRole role = context.partitionRoleFor(node.id());
        if (role == PartitionExecutionRole.INTERIOR) {
            throw new IllegalStateException("Interior CUDA partition node must be covered before prepare: nodeId="
                    + node.id());
        }
        if (role != PartitionExecutionRole.ANCHOR) {
            return new PreparedStepMetadata(
                    ComputeBackend.GPU_CUDA,
                    null,
                    node.inputIds(),
                    CudaDirectPreparedExecutable.prepare(node),
                    InputResidencyRequirement.cpuReadableAll(),
                    OutputResidencyEffect.cpuCurrentPreserveNative()
            );
        }
        LoweredRegion loweredRegion = GpuAcceleratorPrepareSupport.requireLoweredRegion(
                context.cudaLoweredRegionForAnchor(node.id()),
                "CUDA GPU",
                node.id()
        );
        return prepareRegionStep(loweredRegion, context);
    }

    public PreparedStepMetadata prepareRegionStep(
            LoweredRegion loweredRegion,
            BackendPrepareContext context
    ) {
        LoweringFamily loweringFamily = GpuAcceleratorPrepareSupport.resolveLoweringFamily(
                loweredRegion,
                LoweringFamily.CUDA_GRAPH_REGION
        );
        var regionPlan = loweredRegion.units().getFirst().requireRegionPlan();
        RegionPlanValidator.requireBoundaryCoverage(regionPlan, context);
        PartitionPlan genericPlan = context.backendPlanForAnchor(regionPlan.anchorNodeId());
        CudaGpuPartitionPlan plan = GpuAcceleratorPrepareSupport.requirePlan(
                genericPlan,
                CudaGpuPartitionPlan.class,
                "CUDA GPU",
                regionPlan.anchorNodeId()
        );
        var fallback = GpuAcceleratorPrepareSupport.prepareCpuFallback(
                plan,
                context,
                cpuPreparer,
                "CUDA GPU",
                false
        );

        PreparedCudaExecutable executable = new PreparedCudaExecutable(
                plan.dagSpec(),
                loweringFamily,
                regionPlan,
                bridge,
                fallback.preparedSteps(),
                context.runtimeConfig().accelerator().cuda(),
                plan.compoundSummary(),
                plan.manifest()
        );
        return new PreparedStepMetadata(
                ComputeBackend.GPU_CUDA,
                null,
                java.util.List.of(),
                new AcceleratorExecutionArtifact(executable),
                InputResidencyRequirement.none(),
                OutputResidencyEffect.cpuCurrentIfUnset(executable.outputResidencyReason())
        );
    }
}
