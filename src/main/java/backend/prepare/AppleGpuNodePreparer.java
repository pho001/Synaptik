package backend.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.apple.bridge.AppleMpsFfmBridge;
import backend.apple.bridge.AppleMpsGraphBridge;
import backend.apple.exec.PreparedAppleGpuExecutable;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.optimizer.partition.AcceleratorPartitionPlan;
import graph.optimizer.partition.apple.AppleGpuPartitionPlan;

import java.util.ArrayList;
import java.util.List;

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
            return new CompiledNodeExecutionMetadata(ComputeBackend.GPU_METAL, null, null, null, null, null, role);
        }
        if (role != PartitionExecutionRole.ANCHOR) {
            return cpuPreparer.prepareAsCpu(node, context);
        }

        AcceleratorPartitionPlan genericPlan = context.acceleratorPlanForAnchor(node.id());
        if (!(genericPlan instanceof AppleGpuPartitionPlan plan)) {
            throw new IllegalStateException("Missing Apple GPU partition plan for anchor node " + node.id());
        }

        BackendPrepareContext localContext = context.fork();
        List<PreparedAppleGpuExecutable.PreparedStep> preparedSteps = new ArrayList<>(plan.nodeIds().size());
        CompiledNodeExecutionMetadata anchorCpuMetadata = null;
        CompiledNodeExecutionMetadata computeCpuMetadata = null;
        CompiledNode computeNode = null;
        for (int nodeId : plan.nodeIds()) {
            CompiledNode partitionNode = context.compiledNode(nodeId);
            if (partitionNode == null) {
                throw new IllegalStateException("Missing compiled node for Apple partition nodeId=" + nodeId);
            }
            CompiledNodeExecutionMetadata cpuMetadata = cpuPreparer.prepareAsCpu(partitionNode, localContext);
            localContext.publishPreparedMetadata(nodeId, cpuMetadata);
            preparedSteps.add(new PreparedAppleGpuExecutable.PreparedStep(partitionNode, cpuMetadata));
            if (computeNode == null) {
                computeNode = partitionNode;
                computeCpuMetadata = cpuMetadata;
            }
            if (nodeId == plan.anchorNodeId()) {
                anchorCpuMetadata = cpuMetadata;
            }
        }
        if (anchorCpuMetadata == null || anchorCpuMetadata.cpuPlan() == null) {
            throw new IllegalStateException("Missing CPU anchor metadata for Apple partition anchor node " + node.id());
        }
        if (computeNode == null || computeCpuMetadata == null || computeCpuMetadata.cpuPlan() == null) {
            throw new IllegalStateException("Missing CPU compute metadata for Apple partition entry node " + node.id());
        }

        return new CompiledNodeExecutionMetadata(
                ComputeBackend.GPU_METAL,
                null,
                anchorCpuMetadata.cpuPlan(),
                null,
                null,
                new PreparedAppleGpuExecutable(plan, computeNode, computeCpuMetadata.cpuPlan(), bridge, preparedSteps),
                PartitionExecutionRole.ANCHOR
        );
    }
}
