package graph.execution.trace.contrib;

import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.accelerator.lowering.GpuCompoundPatternType;

import java.util.LinkedHashMap;

final class AcceleratorRunTraceContributor implements BackendRunTraceContributor {
    @Override
    public void contribute(BackendRunTraceContext context, LinkedHashMap<String, Object> attrs) {
        var metadata = context.metadata();
        if (metadata.artifact() instanceof AcceleratorExecutionArtifact artifact && artifact.executable() != null) {
            var executable = artifact.executable();
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
            attrs.put("cpuMaterializationCount", context.executionContext().cpuMaterializationTraceCount());
            attrs.put("deviceHandoffCount", deviceHandoffCount(decision));
            var regionPlan = executable.regionExecutionPlan();
            if (regionPlan != null) {
                BackendTraceSupport.addRegionPlanAttrs(attrs, regionPlan);
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
        var layoutTransformDecision = context.executionContext().layoutTransformDecisionForNodeId(context.node().id());
        if (layoutTransformDecision != null) {
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
    }

    private static int deviceHandoffCount(backend.accelerator.buffer.AcceleratorBufferDecision decision) {
        if (decision == null
                || decision.path() != backend.accelerator.buffer.AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return 0;
        }
        return decision.inputs().size() + decision.outputs().size();
    }

    private static boolean isGpuLayoutMaterialization(
            backend.accelerator.buffer.AcceleratorLayoutTransformDecision decision
    ) {
        if (decision == null || !decision.accepted()) {
            return false;
        }
        return decision.kind() == backend.accelerator.buffer.AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION
                || decision.kind() == backend.accelerator.buffer.AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION;
    }
}
