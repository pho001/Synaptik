package backend.accelerator.lowering;

import backend.contract.ComputeBackend;

import java.util.List;
import java.util.Objects;

/**
 * Backend-neutral summary for a compound GPU partition beside the lowered accelerator DAG.
 *
 * <p>The summary is trace and legality metadata. It does not replace the accelerator DAG
 * and must not bypass backend-owned dtype, layout, capability, or native ABI checks.</p>
 */
public record GpuCompoundPartitionSummary(
        ComputeBackend backend,
        GpuCompoundPatternType patternType,
        boolean supported,
        GpuLoweringUnsupportedReason reason,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds,
        List<String> dagNodeTypes,
        List<String> postOps,
        String detail
) {
    public GpuCompoundPartitionSummary {
        backend = backend == null ? ComputeBackend.CPU : backend;
        patternType = patternType == null ? GpuCompoundPatternType.NONE : patternType;
        reason = reason == null ? GpuLoweringUnsupportedReason.COMPOUND_PATTERN_UNSUPPORTED : reason;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        dagNodeTypes = List.copyOf(dagNodeTypes == null ? List.of() : dagNodeTypes);
        postOps = List.copyOf(postOps == null ? List.of() : postOps);
        detail = detail == null ? "" : detail;
        if (supported && reason != GpuLoweringUnsupportedReason.SUPPORTED) {
            throw new IllegalArgumentException("supported compound summary must use SUPPORTED reason");
        }
        if (!supported && reason == GpuLoweringUnsupportedReason.SUPPORTED) {
            throw new IllegalArgumentException("unsupported compound summary must not use SUPPORTED reason");
        }
    }

    /**
     * Returns a non-compound summary.
     */
    public static GpuCompoundPartitionSummary none(ComputeBackend backend, List<Integer> nodeIds) {
        return unsupported(
                backend,
                GpuCompoundPatternType.NONE,
                GpuLoweringUnsupportedReason.COMPOUND_PATTERN_UNSUPPORTED,
                "partition is not a recognized compound GPU pattern",
                nodeIds,
                List.of(),
                List.of(),
                List.of(),
                List.of()
        );
    }

    /**
     * Returns a supported compound summary.
     */
    public static GpuCompoundPartitionSummary supported(
            ComputeBackend backend,
            GpuCompoundPatternType patternType,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<Integer> outputNodeIds,
            List<String> dagNodeTypes,
            List<String> postOps,
            String detail
    ) {
        Objects.requireNonNull(patternType, "patternType cannot be null");
        return new GpuCompoundPartitionSummary(
                backend,
                patternType,
                true,
                GpuLoweringUnsupportedReason.SUPPORTED,
                orderedNodeIds,
                externalInputNodeIds,
                outputNodeIds,
                dagNodeTypes,
                postOps,
                detail
        );
    }

    /**
     * Returns an unsupported compound summary.
     */
    public static GpuCompoundPartitionSummary unsupported(
            ComputeBackend backend,
            GpuCompoundPatternType patternType,
            GpuLoweringUnsupportedReason reason,
            String detail,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<Integer> outputNodeIds,
            List<String> dagNodeTypes,
            List<String> postOps
    ) {
        Objects.requireNonNull(patternType, "patternType cannot be null");
        return new GpuCompoundPartitionSummary(
                backend,
                patternType,
                false,
                reason,
                orderedNodeIds,
                externalInputNodeIds,
                outputNodeIds,
                dagNodeTypes,
                postOps,
                detail
        );
    }
}
