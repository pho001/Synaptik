package backend.accelerator.prepare;

import backend.contract.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu.CpuNodeExecutionArtifact;
import backend.cpu.prepare.CpuNodePreparer;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.compile.planning.partition.PartitionPlan;

import java.util.ArrayList;
import java.util.List;

/**
 * Shared internal SPI for preparing accelerator partitions.
 *
 * <p>CUDA and Metal preparers use these helpers to validate lowered artifacts,
 * assign partition roles, and precompute CPU fallback metadata for every covered
 * node.</p>
 */
public final class GpuAcceleratorPrepareSupport {
    private GpuAcceleratorPrepareSupport() {
    }

    /**
     * Prepared CPU fallback material for an accelerator partition.
     *
     * @param preparedSteps CPU steps in partition order
     * @param computeNode first partition node used by some bridge input rewrites
     * @param computeCpuMetadata CPU metadata for the compute node
     * @param anchorCpuMetadata CPU metadata for the partition anchor node
     */
    public record CpuFallbackPreparation(
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> preparedSteps,
            CompiledNode computeNode,
            CompiledNodeExecutionMetadata computeCpuMetadata,
            CompiledNodeExecutionMetadata anchorCpuMetadata
    ) {
    }

    /**
     * Requires a lowered region to be present for an accelerator anchor node.
     */
    public static LoweredRegion requireLoweredRegion(LoweredRegion loweredRegion, String backendName, int anchorNodeId) {
        if (loweredRegion == null) {
            throw new IllegalStateException("Missing " + backendName + " lowered region for anchor node " + anchorNodeId);
        }
        return loweredRegion;
    }

    /**
     * Resolves the lowering family from a region, using the supplied fallback when none is present.
     */
    public static LoweringFamily resolveLoweringFamily(LoweredRegion loweredRegion, LoweringFamily fallback) {
        if (loweredRegion == null || loweredRegion.units().isEmpty()) {
            return fallback;
        }
        return loweredRegion.units().getFirst().loweringFamily();
    }

    /**
     * Casts a generic partition plan to the backend-specific plan type required by a preparer.
     */
    public static <P extends PartitionPlan> P requirePlan(
            PartitionPlan genericPlan,
            Class<P> planType,
            String backendName,
            int anchorNodeId
    ) {
        if (!planType.isInstance(genericPlan)) {
            throw new IllegalStateException("Missing " + backendName + " partition plan for anchor node " + anchorNodeId);
        }
        return planType.cast(genericPlan);
    }

    /**
     * Prepares CPU execution metadata that can replay a partition when native acceleration is unavailable.
     */
    public static CpuFallbackPreparation prepareCpuFallback(
            PartitionPlan plan,
            BackendPrepareContext context,
            CpuNodePreparer cpuPreparer,
            String backendName,
            boolean requireComputeCpuPlan
    ) {
        BackendPrepareContext localContext = context.fork();
        List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> preparedSteps = new ArrayList<>(plan.nodeIds().size());
        CompiledNode computeNode = null;
        CompiledNodeExecutionMetadata computeCpuMetadata = null;
        CompiledNodeExecutionMetadata anchorCpuMetadata = null;

        for (int nodeId : plan.nodeIds()) {
            CompiledNode partitionNode = context.compiledNode(nodeId);
            if (partitionNode == null) {
                throw new IllegalStateException("Missing compiled node for " + backendName + " partition nodeId=" + nodeId);
            }
            CompiledNodeExecutionMetadata cpuMetadata = cpuPreparer.prepareAsCpu(partitionNode, localContext);
            localContext.publishPreparedMetadata(nodeId, cpuMetadata);
            preparedSteps.add(new PreparedAcceleratorExecutionSupport.CpuFallbackStep(partitionNode, cpuMetadata));
            if (computeNode == null) {
                computeNode = partitionNode;
                computeCpuMetadata = cpuMetadata;
            }
            if (nodeId == plan.anchorNodeId()) {
                anchorCpuMetadata = cpuMetadata;
            }
        }

        CpuNodeExecutionPlan anchorCpuPlan = cpuPlan(anchorCpuMetadata);
        if (anchorCpuPlan == null) {
            throw new IllegalStateException("Missing CPU anchor metadata for " + backendName + " partition anchor node " + plan.anchorNodeId());
        }
        CpuNodeExecutionPlan computeCpuPlan = cpuPlan(computeCpuMetadata);
        if (requireComputeCpuPlan && (computeNode == null || computeCpuPlan == null)) {
            throw new IllegalStateException("Missing CPU compute metadata for " + backendName + " partition entry node " + plan.anchorNodeId());
        }

        return new CpuFallbackPreparation(
                List.copyOf(preparedSteps),
                computeNode,
                computeCpuMetadata,
                anchorCpuMetadata
        );
    }

    private static CpuNodeExecutionPlan cpuPlan(CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            return null;
        }
        if (metadata.artifact() == null) {
            return null;
        }
        return switch (metadata.artifact()) {
            case CpuNodeExecutionArtifact artifact -> artifact.cpuPlan();
            case CpuFusedExecutionArtifact artifact -> artifact.cpuPlan();
            default -> null;
        };
    }
}
