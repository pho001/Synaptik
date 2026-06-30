package trace.prepare;

import java.util.List;
import java.util.Map;

/** Lossless diagnostic snapshot of a backend-owned lowered GPU region manifest. */
public record GpuLoweredRegionTrace(
        String regionId,
        String backend,
        int anchorNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds,
        int selectedRegionLength,
        List<OriginalOperation> originalOperations,
        List<LoweredPrimitive> loweredPrimitives,
        List<ValueAssumption> inputAssumptions,
        List<ValueAssumption> outputAssumptions,
        CompoundSummary compoundSummary,
        List<FusedSubpattern> fusedSubpatterns,
        List<Rejection> rejections,
        CandidateSpan candidateSpan,
        Map<String, String> backendExtensions
) {
    public GpuLoweredRegionTrace {
        regionId = regionId == null ? "" : regionId;
        backend = backend == null ? "" : backend;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        selectedRegionLength = Math.max(0, selectedRegionLength);
        originalOperations = List.copyOf(originalOperations == null ? List.of() : originalOperations);
        loweredPrimitives = List.copyOf(loweredPrimitives == null ? List.of() : loweredPrimitives);
        inputAssumptions = List.copyOf(inputAssumptions == null ? List.of() : inputAssumptions);
        outputAssumptions = List.copyOf(outputAssumptions == null ? List.of() : outputAssumptions);
        compoundSummary = compoundSummary == null ? CompoundSummary.none() : compoundSummary;
        fusedSubpatterns = List.copyOf(fusedSubpatterns == null ? List.of() : fusedSubpatterns);
        rejections = List.copyOf(rejections == null ? List.of() : rejections);
        candidateSpan = candidateSpan == null ? CandidateSpan.empty() : candidateSpan;
        backendExtensions = Map.copyOf(backendExtensions == null ? Map.of() : backendExtensions);
    }

    public record OriginalOperation(
            int nodeId, String opType, List<Integer> inputNodeIds,
            List<Integer> outputNodeIds, String dataType, List<Integer> shape,
            List<String> loweredPrimitiveIds, List<String> reasons
    ) {
        public OriginalOperation {
            opType = opType == null ? "UNKNOWN" : opType;
            inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
            outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
            dataType = dataType == null ? "" : dataType;
            shape = List.copyOf(shape == null ? List.of() : shape);
            loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
            reasons = List.copyOf(reasons == null ? List.of() : reasons);
        }
    }

    public record LoweredPrimitive(
            String primitiveId, String primitiveType, List<Integer> sourceOriginalNodeIds,
            List<String> inputRefs, String outputRef, String dataType,
            List<Integer> shape, List<String> reasons
    ) {
        public LoweredPrimitive {
            primitiveId = primitiveId == null ? "" : primitiveId;
            primitiveType = primitiveType == null ? "UNKNOWN" : primitiveType;
            sourceOriginalNodeIds = List.copyOf(sourceOriginalNodeIds == null ? List.of() : sourceOriginalNodeIds);
            inputRefs = List.copyOf(inputRefs == null ? List.of() : inputRefs);
            outputRef = outputRef == null ? "" : outputRef;
            dataType = dataType == null ? "" : dataType;
            shape = List.copyOf(shape == null ? List.of() : shape);
            reasons = List.copyOf(reasons == null ? List.of() : reasons);
        }
    }

    public record ValueAssumption(
            int nodeId, String role, String dataType, int rank, List<Integer> shape,
            String layout, boolean contiguous, boolean hasStorageOffset, long storageOffset
    ) {
        public ValueAssumption {
            role = role == null ? "UNKNOWN" : role;
            dataType = dataType == null ? "" : dataType;
            rank = Math.max(0, rank);
            shape = List.copyOf(shape == null ? List.of() : shape);
            layout = layout == null ? "UNKNOWN" : layout;
            storageOffset = Math.max(0L, storageOffset);
        }
    }

    public record CompoundSummary(
            String backend, String patternType, boolean supported, String reason,
            List<Integer> orderedNodeIds, List<Integer> externalInputNodeIds,
            List<Integer> outputNodeIds, List<String> dagNodeTypes,
            List<String> postOps, String detail
    ) {
        public CompoundSummary {
            backend = backend == null ? "" : backend;
            patternType = patternType == null ? "NONE" : patternType;
            reason = reason == null ? "" : reason;
            orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
            externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
            outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
            dagNodeTypes = List.copyOf(dagNodeTypes == null ? List.of() : dagNodeTypes);
            postOps = List.copyOf(postOps == null ? List.of() : postOps);
            detail = detail == null ? "" : detail;
        }

        public static CompoundSummary none() {
            return new CompoundSummary("", "NONE", false, "", List.of(), List.of(),
                    List.of(), List.of(), List.of(), "");
        }
    }

    public record FusedSubpattern(
            String patternType, boolean supported, List<Integer> originalOperationNodeIds,
            List<String> loweredPrimitiveIds, int loweredPrimitiveCount,
            String reason, String detail
    ) {
        public FusedSubpattern {
            patternType = patternType == null ? "NONE" : patternType;
            originalOperationNodeIds = List.copyOf(
                    originalOperationNodeIds == null ? List.of() : originalOperationNodeIds
            );
            loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
            loweredPrimitiveCount = Math.max(0, loweredPrimitiveCount);
            reason = reason == null ? "" : reason;
            detail = detail == null ? "" : detail;
        }
    }

    public record Rejection(
            String level, int originalNodeId, String primitiveId,
            String fusedPatternType, String reason, String detail
    ) {
        public Rejection {
            level = level == null ? "UNKNOWN" : level;
            primitiveId = primitiveId == null ? "" : primitiveId;
            fusedPatternType = fusedPatternType == null ? "" : fusedPatternType;
            reason = reason == null ? "" : reason;
            detail = detail == null ? "" : detail;
        }
    }

    public record CandidateSpan(
            List<Integer> originalCandidateNodeIds, List<Integer> acceptedNodeIds,
            int rejectedOriginalNodeId, String rejectedPrimitiveId, String reason
    ) {
        public CandidateSpan {
            originalCandidateNodeIds = List.copyOf(
                    originalCandidateNodeIds == null ? List.of() : originalCandidateNodeIds
            );
            acceptedNodeIds = List.copyOf(acceptedNodeIds == null ? List.of() : acceptedNodeIds);
            rejectedPrimitiveId = rejectedPrimitiveId == null ? "" : rejectedPrimitiveId;
            reason = reason == null ? "" : reason;
        }

        public static CandidateSpan empty() {
            return new CandidateSpan(List.of(), List.of(), -1, "", "");
        }
    }
}
