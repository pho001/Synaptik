package backend.accelerator.exec;

import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.accelerator.buffer.AcceleratorLayoutTransformKind;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.lowering.region.RegionExecutionPlan;
import backend.runtime.ExecutionContext;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedExecutionArtifact;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import graph.execution.trace.StepTraceContribution;

import java.util.LinkedHashMap;

public record AcceleratorExecutionArtifact(
        PreparedAcceleratorExecutable executable
) implements PreparedExecutionArtifact {
    @Override
    public void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
        if (executable == null) {
            return;
        }
        for (var fallbackStep : executable.cpuFallbackSteps()) {
            if (fallbackStep != null && fallbackStep.metadata().artifact() != null) {
                fallbackStep.metadata().artifact().allocateRuntimeState(fallbackStep.node().id(), allocator);
            }
        }
    }

    @Override
    public StepTraceContribution traceContribution(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        if (executable != null) {
            var decision = executable.lastAcceleratorBufferDecision();
            attrs.put("acceleratorBufferMode", decision.mode().name());
            attrs.put("acceleratorBufferBackend", decision.backend().name());
            attrs.put("acceleratorBufferDecision", decision.path().name());
            attrs.put("acceleratorBufferExecutionPath", decision.path().name());
            attrs.put("acceleratorBufferReasonCode", decision.reasonCode().name());
            attrs.put("acceleratorBufferReason", decision.reason());
            attrs.put("acceleratorBufferPreparedInputUsed", decision.preparedInputUsed());
            attrs.put("acceleratorBufferInputCount", decision.inputs().size());
            attrs.put("acceleratorBufferOutputCount", decision.outputs().size());
            attrs.put("cpuMaterializationCount", context.cpuMaterializationTraceCount());
            attrs.put("deviceHandoffCount", decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING
                    ? decision.inputs().size() + decision.outputs().size()
                    : 0);
            RegionExecutionPlan regionPlan = executable.regionExecutionPlan();
            if (regionPlan != null) {
                addRegionPlanAttrs(attrs, regionPlan);
            }
            var manifest = executable.gpuLoweredRegionManifest();
            if (manifest != null && !manifest.regionId().isBlank()) {
                attrs.put("gpuRegionId", manifest.regionId());
                attrs.put("gpuLoweredRegionId", manifest.regionId());
            }
            if (manifest != null) {
                attrs.put("selectedRegionLength", manifest.selectedRegionLength());
                attrs.put("loweredPrimitiveCount", manifest.loweredPrimitives().size());
                attrs.put("gpuFusedSubpatternCount", manifest.fusedSubpatterns().size());
                attrs.put("gpuFusedSubpatternTypes", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.patternType().name())
                        .toList());
            }
            if (manifest != null && !manifest.fusedSubpatterns().isEmpty()) {
                attrs.put("gpuFusedSubpatternOriginalNodeIds", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.originalOperationNodeIds())
                        .toList());
                attrs.put("gpuFusedSubpatternLoweredPrimitiveCount", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.loweredPrimitiveCount())
                        .toList());
                attrs.put("gpuFusedSubpatternReasons", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.reason().name())
                        .toList());
            }
            var compoundSummary = executable.compoundSummary();
            if (compoundSummary != null && compoundSummary.patternType() != GpuCompoundPatternType.NONE) {
                attrs.put("gpuCompoundPattern", compoundSummary.patternType().name());
                attrs.put("gpuCompoundSupported", compoundSummary.supported());
                attrs.put("gpuCompoundReason", compoundSummary.reason().name());
                attrs.put("gpuCompoundNodeCount", compoundSummary.orderedNodeIds().size());
                attrs.put("gpuCompoundOrderedNodeIds", compoundSummary.orderedNodeIds());
                attrs.put("gpuCompoundDagNodeTypes", compoundSummary.dagNodeTypes());
                attrs.put("gpuCompoundPostOps", compoundSummary.postOps());
            }
            executable.contributeRunTraceAttributes(attrs);
        }
        addLayoutTransformAttrs(node, context, attrs);
        return new StepTraceContribution("", attrs, null, null, null, null, null, null, null);
    }

    private static void addLayoutTransformAttrs(
            CompiledNode node,
            ExecutionContext context,
            LinkedHashMap<String, Object> attrs
    ) {
        AcceleratorLayoutTransformDecision layoutTransformDecision = context.layoutTransformDecisionForNodeId(node.id());
        if (layoutTransformDecision == null) {
            return;
        }
        attrs.put("gpuLayoutTransformKind", layoutTransformDecision.kind().name());
        attrs.put("gpuLayoutTransformOp", layoutTransformDecision.opType().name());
        attrs.put("gpuLayoutTransformSourceNodeId", layoutTransformDecision.sourceNodeId());
        attrs.put("gpuLayoutTransformTargetNodeId", layoutTransformDecision.targetNodeId());
        attrs.put("gpuLayoutTransformAccepted", layoutTransformDecision.accepted());
        attrs.put("gpuLayoutTransformReasonCode", layoutTransformDecision.reasonCode().name());
        attrs.put("gpuLayoutTransformReason", layoutTransformDecision.reason());
        attrs.put("gpuLayoutTransformSourceLayoutClass", layoutTransformDecision.sourceLayout().layoutClass().name());
        attrs.put("gpuLayoutTransformTargetLayoutClass", layoutTransformDecision.targetLayout().layoutClass().name());
        attrs.put("gpuLayoutTransformBytes", layoutTransformDecision.targetLayout().logicalByteLength());
        attrs.put("gpuLayoutMaterializationCount", isGpuLayoutMaterialization(layoutTransformDecision) ? 1 : 0);
        attrs.put("gpuLayoutMaterializationBytes",
                isGpuLayoutMaterialization(layoutTransformDecision)
                        ? layoutTransformDecision.targetLayout().logicalByteLength()
                        : 0L);
        attrs.putIfAbsent("acceleratorBufferBackend", layoutTransformDecision.backendId());
        attrs.putIfAbsent("acceleratorBufferDecision", layoutTransformDecision.kind().name());
        attrs.putIfAbsent("acceleratorBufferExecutionPath", layoutTransformDecision.accepted()
                ? layoutTransformDecision.kind().name()
                : "UNAVAILABLE");
        attrs.putIfAbsent("acceleratorBufferReasonCode", layoutTransformDecision.reasonCode().name());
        attrs.putIfAbsent("acceleratorBufferReason", layoutTransformDecision.reason());
    }

    private static boolean isGpuLayoutMaterialization(AcceleratorLayoutTransformDecision decision) {
        if (decision == null || !decision.accepted()) {
            return false;
        }
        return decision.kind() == AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION
                || decision.kind() == AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION;
    }

    private static void addRegionPlanAttrs(LinkedHashMap<String, Object> attrs, RegionExecutionPlan regionPlan) {
        attrs.put("regionId", regionPlan.regionId());
        attrs.put("regionTarget", regionPlan.target().name());
        attrs.put("loweringFamily", regionPlan.loweringFamily().name());
        attrs.put("anchorNodeId", regionPlan.anchorNodeId());
        attrs.put("orderedNodeIds", regionPlan.orderedNodeIds());
        attrs.put("boundaryOutputNodeIds", regionPlan.boundaryOutputNodeIds());
        attrs.put("regionNodeCount", regionPlan.orderedNodeIds().size());
        attrs.put("regionDecision", regionPlan.decision().selected() ? "SELECTED" : "REJECTED");
        attrs.put("regionReason", regionPlan.decision().reason());
        attrs.put("regionExecutionKindSummary", regionPlan.executionGroups().stream()
                .map(group -> group.executionKind().name())
                .distinct()
                .toList());
        attrs.put("regionStorageContractSummary", regionPlan.executionGroups().stream()
                .map(group -> group.storageContract().name())
                .distinct()
                .toList());
    }
}
