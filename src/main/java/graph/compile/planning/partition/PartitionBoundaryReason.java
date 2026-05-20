package graph.compile.planning.partition;

/**
 * Normalized reason why partition growth stopped or a candidate was rejected.
 */
public enum PartitionBoundaryReason {
    NONE,
    COVERED_BY_EARLIER_PARTITION,
    UNSUPPORTED_START_NODE,
    MISSING_STRUCTURAL_CANDIDATE,
    LOWERER_REJECTED,
    UNSUPPORTED_NODE,
    EXTERNAL_INPUT_NOT_ALLOWED,
    PRODUCER_CLOSURE_PHASE_BOUNDARY,
    CONSUMER_CLOSURE_PHASE_BOUNDARY,
    MISSING_NODE,
    MISSING_INPUT_NODE,
    MAX_SEARCH_NODES,
    FRONTIER_EXHAUSTED,
    BUDGET_STOP,
    UNKNOWN;

    /**
     * Converts planner trace text to an enum value.
     *
     * @param reason planner reason string
     * @return normalized boundary reason
     */
    public static PartitionBoundaryReason fromReason(String reason) {
        if (reason == null || reason.isBlank()) {
            return NONE;
        }
        if (reason.startsWith("anchor-")) {
            reason = reason.substring("anchor-".length());
        }
        return switch (reason) {
            case "covered-by-earlier-partition" -> COVERED_BY_EARLIER_PARTITION;
            case "unsupported-start-node" -> UNSUPPORTED_START_NODE;
            case "missing-structural-candidate" -> MISSING_STRUCTURAL_CANDIDATE;
            case "lowerer-rejected" -> LOWERER_REJECTED;
            case "unsupported-node" -> UNSUPPORTED_NODE;
            case "external-input-not-allowed" -> EXTERNAL_INPUT_NOT_ALLOWED;
            case "producer-closure-phase-boundary" -> PRODUCER_CLOSURE_PHASE_BOUNDARY;
            case "consumer-closure-phase-boundary" -> CONSUMER_CLOSURE_PHASE_BOUNDARY;
            case "missing-node" -> MISSING_NODE;
            case "missing-input-node" -> MISSING_INPUT_NODE;
            case "max-search-nodes" -> MAX_SEARCH_NODES;
            case "frontier-exhausted" -> FRONTIER_EXHAUSTED;
            case "budget-stop" -> BUDGET_STOP;
            default -> UNKNOWN;
        };
    }
}
