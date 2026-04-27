package backend.accelerator.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.cpu.prepare.CpuNodePreparer;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.optimizer.partition.PartitionPlan;

import java.util.ArrayList;
import java.util.List;

public final class GpuAcceleratorPrepareSupport {
    private GpuAcceleratorPrepareSupport() {
    }

    public record CpuFallbackPreparation(
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> preparedSteps,
            CompiledNode computeNode,
            CompiledNodeExecutionMetadata computeCpuMetadata,
            CompiledNodeExecutionMetadata anchorCpuMetadata
    ) {
    }

    public static CompiledNodeExecutionMetadata interiorMetadata(ComputeBackend backend, PartitionExecutionRole role) {
        return new CompiledNodeExecutionMetadata(backend, null, null, null, null, null, role);
    }

    public static LoweredRegion requireLoweredRegion(LoweredRegion loweredRegion, String backendName, int anchorNodeId) {
        if (loweredRegion == null) {
            throw new IllegalStateException("Missing " + backendName + " lowered region for anchor node " + anchorNodeId);
        }
        return loweredRegion;
    }

    public static LoweringFamily resolveLoweringFamily(LoweredRegion loweredRegion, LoweringFamily fallback) {
        if (loweredRegion == null || loweredRegion.units().isEmpty()) {
            return fallback;
        }
        return loweredRegion.units().getFirst().loweringFamily();
    }

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

        CpuNodeExecutionPlan anchorCpuPlan = anchorCpuMetadata == null ? null : anchorCpuMetadata.cpuPlan();
        if (anchorCpuPlan == null) {
            throw new IllegalStateException("Missing CPU anchor metadata for " + backendName + " partition anchor node " + plan.anchorNodeId());
        }
        CpuNodeExecutionPlan computeCpuPlan = computeCpuMetadata == null ? null : computeCpuMetadata.cpuPlan();
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
}
