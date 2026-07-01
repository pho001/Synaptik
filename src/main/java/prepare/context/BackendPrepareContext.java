package prepare.context;

import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredPartition;
import config.runtime.RuntimeConfig;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;
import runtime.execution.PreparedStepMetadata;
import planning.partition.PartitionPlan;

import java.util.List;
import java.util.Map;

public final class BackendPrepareContext {
    private final PrepareInputs inputs;
    private final PreparedMetadataIndex metadataIndex;
    private final BackendPlanIndex backendPlanIndex;
    private final PartitionRoleIndex roleIndex;
    private final LoweredPartitionIndex loweredPartitionIndex;

    public BackendPrepareContext(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Integer, List<CompiledNode>> consumers
    ) {
        this(
                new PrepareInputs(runtimeConfig, supportsBackward, compiledNodes, descriptorIndex, consumers),
                new PreparedMetadataIndex(),
                new BackendPlanIndex(),
                new PartitionRoleIndex(),
                new LoweredPartitionIndex()
        );
    }

    private BackendPrepareContext(
            PrepareInputs inputs,
            PreparedMetadataIndex metadataIndex,
            BackendPlanIndex backendPlanIndex,
            PartitionRoleIndex roleIndex,
            LoweredPartitionIndex loweredPartitionIndex
    ) {
        this.inputs = inputs;
        this.metadataIndex = metadataIndex;
        this.backendPlanIndex = backendPlanIndex;
        this.roleIndex = roleIndex;
        this.loweredPartitionIndex = loweredPartitionIndex;
    }

    public RuntimeConfig runtimeConfig() {
        return inputs.runtimeConfig();
    }

    public boolean supportsBackward() {
        return inputs.supportsBackward();
    }

    public List<CompiledNode> compiledNodes() {
        return inputs.compiledNodes();
    }

    public CompiledNode compiledNode(int nodeId) {
        return inputs.compiledNode(nodeId);
    }

    public CompiledTensorDescriptor descriptor(int nodeId) {
        return inputs.descriptor(nodeId);
    }

    public CompiledTensorDescriptorIndex descriptorIndex() {
        return inputs.descriptorIndex();
    }

    public List<CompiledNode> consumersFor(int nodeId) {
        return inputs.consumersFor(nodeId);
    }

    public PreparedStepMetadata preparedMetadataFor(int nodeId) {
        return metadataIndex.metadataFor(nodeId);
    }

    public void publishPreparedMetadata(int nodeId, PreparedStepMetadata metadata) {
        metadataIndex.publish(nodeId, metadata);
    }

    public void publishBackendPlans(List<PartitionPlan> plans) {
        backendPlanIndex.publish(plans, roleIndex);
    }

    public PartitionPlan backendPlanForAnchor(int nodeId) {
        return backendPlanIndex.planForAnchor(nodeId);
    }

    public void publishLoweredPartitions(List<LoweredPartition> loweredPartitions) {
        loweredPartitionIndex.publish(loweredPartitions, roleIndex);
    }

    public LoweredExecutionUnit cpuLoweredUnitForAnchor(int nodeId) {
        return loweredPartitionIndex.cpuUnitForAnchor(nodeId);
    }

    public LoweredExecutionUnit cpuFusedUnitForStart(int nodeId) {
        return loweredPartitionIndex.cpuFusedUnitForStart(nodeId);
    }

    public LoweredExecutionUnit cpuSpecializedUnitForStart(int nodeId) {
        return loweredPartitionIndex.cpuSpecializedUnitForStart(nodeId);
    }

    public LoweredPartition metalLoweredPartitionForAnchor(int nodeId) {
        return loweredPartitionIndex.metalPartitionForAnchor(nodeId);
    }

    public LoweredPartition metalLoweredPartitionForStart(int nodeId) {
        return loweredPartitionIndex.metalPartitionForStart(nodeId);
    }

    public LoweredPartition cudaLoweredPartitionForAnchor(int nodeId) {
        return loweredPartitionIndex.cudaPartitionForAnchor(nodeId);
    }

    public LoweredPartition cudaLoweredPartitionForStart(int nodeId) {
        return loweredPartitionIndex.cudaPartitionForStart(nodeId);
    }

    public PartitionExecutionRole partitionRoleFor(int nodeId) {
        return roleIndex.roleFor(nodeId);
    }

    public BackendPrepareContext fork() {
        return new BackendPrepareContext(
                inputs,
                metadataIndex.fork(),
                backendPlanIndex.fork(),
                roleIndex.fork(),
                loweredPartitionIndex.fork()
        );
    }
}
