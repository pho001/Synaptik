package backend.accelerator.lowering;

/**
 * Stable compact text renderer for lowered GPU partition manifests.
 */
public final class GpuLoweredPartitionManifestRenderer {
    private GpuLoweredPartitionManifestRenderer() {
    }

    /**
     * Renders a deterministic multi-line manifest summary for human-readable traces and reports.
     *
     * @param manifest lowered GPU partition manifest
     * @return compact manifest block, or an empty string when the manifest is absent
     */
    public static String renderCompact(GpuLoweredPartitionManifest manifest) {
        if (manifest == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("GPU Lowered Partition\n");
        sb.append("partitionId: ").append(manifest.partitionId()).append('\n');
        sb.append("backend: ").append(manifest.backend()).append('\n');
        sb.append("selectedPartitionLength: ").append(manifest.selectedPartitionLength()).append('\n');
        sb.append("anchorNodeId: ").append(manifest.anchorNodeId()).append('\n');
        sb.append("orderedNodeIds: ").append(manifest.orderedNodeIds()).append('\n');
        sb.append("externalInputNodeIds: ").append(manifest.externalInputNodeIds()).append('\n');
        sb.append("outputNodeIds: ").append(manifest.outputNodeIds()).append('\n');

        sb.append("Original Ops\n");
        if (manifest.originalOps().isEmpty()) {
            sb.append("- none\n");
        } else {
            for (GpuLoweredPartitionOriginalOp op : manifest.originalOps()) {
                sb.append("- nodeId=").append(op.nodeId())
                        .append(" opType=").append(op.opType())
                        .append(" inputs=").append(op.inputNodeIds())
                        .append(" outputs=").append(op.outputNodeIds())
                        .append(" dtype=").append(op.dataType())
                        .append(" shape=").append(op.shape())
                        .append(" primitives=").append(op.loweredPrimitiveIds())
                        .append(" reasons=").append(op.aggregatedReasons())
                        .append('\n');
            }
        }

        sb.append("Lowered Primitives\n");
        if (manifest.loweredPrimitives().isEmpty()) {
            sb.append("- none\n");
        } else {
            for (GpuLoweredPrimitiveManifest primitive : manifest.loweredPrimitives()) {
                sb.append("- primitiveId=").append(primitive.primitiveId())
                        .append(" primitiveType=").append(primitive.primitiveType())
                        .append(" sourceOriginalNodeIds=").append(primitive.sourceOriginalNodeIds())
                        .append(" inputRefs=").append(primitive.inputRefs())
                        .append(" outputRef=").append(primitive.outputRef())
                        .append(" dtype=").append(primitive.dataType())
                        .append(" shape=").append(primitive.shape())
                        .append(" reasons=").append(primitive.reasons())
                        .append('\n');
            }
        }

        sb.append("Value Assumptions\n");
        appendAssumptions(sb, "input", manifest.inputAssumptions());
        appendAssumptions(sb, "output", manifest.outputAssumptions());

        sb.append("DType Residency\n");
        boolean dtypeResidencyRendered = false;
        for (var entry : manifest.backendExtensions().entrySet()) {
            if (entry.getKey().startsWith("dtypeResidency.")) {
                sb.append("- ").append(entry.getKey()).append('=').append(entry.getValue()).append('\n');
                dtypeResidencyRendered = true;
            }
        }
        for (GpuLoweredPartitionRejection rejection : manifest.rejections()) {
            if (rejection.detail().contains("dtypeResidency")) {
                sb.append("- rejection level=").append(rejection.level())
                        .append(" originalNodeId=").append(rejection.originalNodeId())
                        .append(" primitiveId=").append(rejection.primitiveId())
                        .append(" reason=").append(rejection.reason())
                        .append(" detail=").append(rejection.detail())
                        .append('\n');
                dtypeResidencyRendered = true;
            }
        }
        if (!dtypeResidencyRendered) {
            sb.append("- none\n");
        }

        sb.append("Fused Subpatterns\n");
        if (manifest.fusedSubpatterns().isEmpty()) {
            GpuCompoundPartitionSummary fused = manifest.fusedSummary();
            if (fused == null) {
                sb.append("- none\n");
            } else {
                sb.append("- patternType=").append(fused.patternType())
                        .append(" supported=").append(fused.supported())
                        .append(" reason=").append(fused.reason())
                        .append(" orderedNodeIds=").append(fused.orderedNodeIds())
                        .append(" dagNodeTypes=").append(fused.dagNodeTypes())
                        .append(" postOps=").append(fused.postOps())
                        .append(" detail=").append(fused.detail())
                        .append('\n');
            }
        } else {
            for (GpuFusionSubpatternSummary subpattern : manifest.fusedSubpatterns()) {
                sb.append("- patternType=").append(subpattern.patternType())
                        .append(" supported=").append(subpattern.supported())
                        .append(" reason=").append(subpattern.reason())
                        .append(" originalOperationNodeIds=").append(subpattern.originalOperationNodeIds())
                        .append(" loweredPrimitiveIds=").append(subpattern.loweredPrimitiveIds())
                        .append(" loweredPrimitiveCount=").append(subpattern.loweredPrimitiveCount())
                        .append(" detail=").append(subpattern.detail())
                        .append('\n');
            }
        }

        sb.append("Rejections\n");
        if (manifest.rejections().isEmpty()) {
            sb.append("- none\n");
        } else {
            for (GpuLoweredPartitionRejection rejection : manifest.rejections()) {
                sb.append("- level=").append(rejection.level())
                        .append(" originalNodeId=").append(rejection.originalNodeId())
                        .append(" primitiveId=").append(rejection.primitiveId())
                        .append(" fusedPatternType=").append(rejection.fusedPatternType())
                        .append(" reason=").append(rejection.reason())
                        .append(" detail=").append(rejection.detail())
                        .append('\n');
            }
        }
        sb.append("candidateSpan: original=")
                .append(manifest.candidateSpan().originalCandidateNodeIds())
                .append(" accepted=").append(manifest.candidateSpan().acceptedNodeIds())
                .append(" rejectedOriginalNodeId=").append(manifest.candidateSpan().rejectedOriginalNodeId())
                .append(" rejectedPrimitiveId=").append(manifest.candidateSpan().rejectedPrimitiveId())
                .append(" reason=").append(manifest.candidateSpan().reason())
                .append('\n');
        return sb.toString();
    }

    private static void appendAssumptions(
            StringBuilder sb,
            String prefix,
            java.util.List<GpuLoweredPartitionValueAssumption> assumptions
    ) {
        if (assumptions == null || assumptions.isEmpty()) {
            sb.append("- ").append(prefix).append(": none\n");
            return;
        }
        for (GpuLoweredPartitionValueAssumption assumption : assumptions) {
            sb.append("- ").append(prefix)
                    .append(" nodeId=").append(assumption.nodeId())
                    .append(" role=").append(assumption.role())
                    .append(" dtype=").append(assumption.dataType())
                    .append(" rank=").append(assumption.rank())
                    .append(" shape=").append(assumption.shape())
                    .append(" layout=").append(assumption.layout())
                    .append(" contiguous=").append(assumption.contiguous())
                    .append(" hasStorageOffset=").append(assumption.hasStorageOffset())
                    .append(" storageOffset=").append(assumption.storageOffset())
                    .append('\n');
        }
    }
}
