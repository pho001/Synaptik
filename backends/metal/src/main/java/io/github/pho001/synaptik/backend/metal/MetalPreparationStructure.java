package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.id.TracePreparedUnitId;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.util.List;
import java.util.Objects;

/** Bounded, data-free structural facts for one finalized Metal preparation plan. */
public record MetalPreparationStructure(
        TracePreparedUnitId preparedUnitId,
        TraceNumericalProfile profile,
        TraceRouteKind route,
        AnchorFamily anchorFamily,
        AnchorDisposition anchorDisposition,
        boolean scalarMultiply,
        boolean externalAdd,
        EpilogueOrder epilogueOrder,
        EpilogueTerminal terminal,
        int anchorCount,
        int anchorMemberCount,
        int schemaVersion,
        int generatorVersion,
        int valueCount,
        int nodeCount,
        int feedCount,
        int targetCount,
        int internalCount,
        int stepCount,
        int instructionCount,
        String canonicalDigest,
        List<CustomStepSummary> plannedCustomSteps,
        boolean plannedCustomStepsTruncated) implements TracePayload {
    /** Maximum summaries exposed by one structural event. */
    public static final int MAX_PLANNED_CUSTOM_STEPS = 32;

    public MetalPreparationStructure {
        Objects.requireNonNull(preparedUnitId, "preparedUnitId");
        Objects.requireNonNull(profile, "profile");
        Objects.requireNonNull(route, "route");
        Objects.requireNonNull(anchorFamily, "anchorFamily");
        Objects.requireNonNull(anchorDisposition, "anchorDisposition");
        Objects.requireNonNull(epilogueOrder, "epilogueOrder");
        Objects.requireNonNull(terminal, "terminal");
        Objects.requireNonNull(canonicalDigest, "canonicalDigest");
        Objects.requireNonNull(plannedCustomSteps, "plannedCustomSteps");
        if (!canonicalDigest.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("canonicalDigest must be lowercase SHA-256");
        }
        if (plannedCustomSteps.size() > MAX_PLANNED_CUSTOM_STEPS) {
            throw new IllegalArgumentException("too many plannedCustomSteps");
        }
        plannedCustomSteps = List.copyOf(plannedCustomSteps);
        requireNonNegative(
                anchorCount,
                anchorMemberCount,
                schemaVersion,
                generatorVersion,
                valueCount,
                nodeCount,
                feedCount,
                targetCount,
                internalCount,
                stepCount,
                instructionCount);
    }

    private static void requireNonNegative(int... values) {
        for (int value : values) {
            if (value < 0) throw new IllegalArgumentException("counts must be non-negative");
        }
    }

    /** Anchor operation family represented by the plan. */
    public enum AnchorFamily {
        NONE,
        MATMUL,
        CONV2D,
        MIXED
    }

    /** Whether recognized anchor work remains composed or executes as a fused step. */
    public enum AnchorDisposition {
        NONE,
        COMPOSED,
        FUSED,
        MIXED
    }

    /** Exact bounded suffix order for a single anchor, or {@link #MULTIPLE}. */
    public enum EpilogueOrder {
        NONE,
        SCALAR,
        ADD,
        RELU,
        CLAMP,
        SCALAR_ADD,
        SCALAR_RELU,
        SCALAR_CLAMP,
        ADD_RELU,
        ADD_CLAMP,
        SCALAR_ADD_RELU,
        SCALAR_ADD_CLAMP,
        MULTIPLE
    }

    /** Terminal operation represented by a single anchor suffix. */
    public enum EpilogueTerminal {
        NONE,
        RELU,
        CLAMP,
        MULTIPLE
    }

    /** Neutral deterministic custom-plan step kind. */
    public enum CustomStepKind {
        FIXED_CUSTOM,
        GRAPH_BOUNDARY,
        GENERATED_POINTWISE,
        ANCHOR_EPILOGUE
    }

    /** Bounded planned-step fact; it never claims submission or completion. */
    public record CustomStepSummary(
            int ordinal,
            CustomStepKind kind,
            AnchorFamily anchorFamily,
            int memberCount,
            int instructionCount) {
        public CustomStepSummary {
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(anchorFamily, "anchorFamily");
            if (ordinal < 0 || memberCount <= 0 || instructionCount < 0) {
                throw new IllegalArgumentException("invalid planned custom-step summary");
            }
        }
    }
}
