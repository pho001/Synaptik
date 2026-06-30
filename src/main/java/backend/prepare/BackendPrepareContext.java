package backend.prepare;

import backend.accelerator.exec.PartitionExecutionRole;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredRegion;
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
    private final LoweredRegionIndex loweredRegionIndex;

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
                new LoweredRegionIndex()
        );
    }

    private BackendPrepareContext(
            PrepareInputs inputs,
            PreparedMetadataIndex metadataIndex,
            BackendPlanIndex backendPlanIndex,
            PartitionRoleIndex roleIndex,
            LoweredRegionIndex loweredRegionIndex
    ) {
        this.inputs = inputs;
        this.metadataIndex = metadataIndex;
        this.backendPlanIndex = backendPlanIndex;
        this.roleIndex = roleIndex;
        this.loweredRegionIndex = loweredRegionIndex;
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

    public void publishLoweredRegions(List<LoweredRegion> loweredRegions) {
        loweredRegionIndex.publish(loweredRegions, roleIndex);
    }

    public LoweredExecutionUnit cpuLoweredUnitForAnchor(int nodeId) {
        return loweredRegionIndex.cpuUnitForAnchor(nodeId);
    }

    public LoweredExecutionUnit cpuFusedUnitForStart(int nodeId) {
        return loweredRegionIndex.cpuFusedUnitForStart(nodeId);
    }

    public LoweredExecutionUnit cpuSpecializedUnitForStart(int nodeId) {
        return loweredRegionIndex.cpuSpecializedUnitForStart(nodeId);
    }

    public LoweredRegion metalLoweredRegionForAnchor(int nodeId) {
        return loweredRegionIndex.metalRegionForAnchor(nodeId);
    }

    public LoweredRegion metalLoweredRegionForStart(int nodeId) {
        return loweredRegionIndex.metalRegionForStart(nodeId);
    }

    public LoweredRegion cudaLoweredRegionForAnchor(int nodeId) {
        return loweredRegionIndex.cudaRegionForAnchor(nodeId);
    }

    public LoweredRegion cudaLoweredRegionForStart(int nodeId) {
        return loweredRegionIndex.cudaRegionForStart(nodeId);
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
                loweredRegionIndex.fork()
        );
    }
}
