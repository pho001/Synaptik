package testsupport;

import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.accelerator.lowering.GpuLoweredRegionValueAssumption;
import trace.prepare.GpuLoweredRegionTrace;

/** Test fixture conversion for synthetic backend manifests used as trace inputs. */
public final class TraceSnapshotTestSupport {
    private TraceSnapshotTestSupport() {
    }

    public static GpuLoweredRegionTrace traceManifest(GpuLoweredRegionManifest source) {
        if (source == null) {
            return null;
        }
        return new GpuLoweredRegionTrace(
                source.regionId(), source.backend().name(), source.anchorNodeId(), source.orderedNodeIds(),
                source.externalInputNodeIds(), source.outputNodeIds(), source.selectedRegionLength(),
                source.originalOps().stream().map(op -> new GpuLoweredRegionTrace.OriginalOperation(
                        op.nodeId(), op.opType(), op.inputNodeIds(), op.outputNodeIds(), op.dataType().name(),
                        op.shape(), op.loweredPrimitiveIds(), op.aggregatedReasons().stream().map(Enum::name).toList()
                )).toList(),
                source.loweredPrimitives().stream().map(primitive -> new GpuLoweredRegionTrace.LoweredPrimitive(
                        primitive.primitiveId(), primitive.primitiveType(), primitive.sourceOriginalNodeIds(),
                        primitive.inputRefs(), primitive.outputRef(), primitive.dataType().name(), primitive.shape(),
                        primitive.reasons().stream().map(Enum::name).toList()
                )).toList(),
                source.inputAssumptions().stream().map(TraceSnapshotTestSupport::traceAssumption).toList(),
                source.outputAssumptions().stream().map(TraceSnapshotTestSupport::traceAssumption).toList(),
                new GpuLoweredRegionTrace.CompoundSummary(
                        source.fusedSummary().backend().name(), source.fusedSummary().patternType().name(),
                        source.fusedSummary().supported(), source.fusedSummary().reason().name(),
                        source.fusedSummary().orderedNodeIds(), source.fusedSummary().externalInputNodeIds(),
                        source.fusedSummary().outputNodeIds(), source.fusedSummary().dagNodeTypes(),
                        source.fusedSummary().postOps(), source.fusedSummary().detail()
                ),
                source.fusedSubpatterns().stream().map(summary -> new GpuLoweredRegionTrace.FusedSubpattern(
                        summary.patternType().name(), summary.supported(), summary.originalOperationNodeIds(),
                        summary.loweredPrimitiveIds(), summary.loweredPrimitiveCount(), summary.reason().name(),
                        summary.detail()
                )).toList(),
                source.rejections().stream().map(rejection -> new GpuLoweredRegionTrace.Rejection(
                        rejection.level(), rejection.originalNodeId(), rejection.primitiveId(),
                        rejection.fusedPatternType(), rejection.reason().name(), rejection.detail()
                )).toList(),
                new GpuLoweredRegionTrace.CandidateSpan(
                        source.candidateSpan().originalCandidateNodeIds(), source.candidateSpan().acceptedNodeIds(),
                        source.candidateSpan().rejectedOriginalNodeId(), source.candidateSpan().rejectedPrimitiveId(),
                        source.candidateSpan().reason().name()
                ),
                source.backendExtensions()
        );
    }

    private static GpuLoweredRegionTrace.ValueAssumption traceAssumption(GpuLoweredRegionValueAssumption source) {
        return new GpuLoweredRegionTrace.ValueAssumption(
                source.nodeId(), source.role(), source.dataType().name(), source.rank(), source.shape(),
                source.layout(), source.contiguous(), source.hasStorageOffset(), source.storageOffset()
        );
    }
}
